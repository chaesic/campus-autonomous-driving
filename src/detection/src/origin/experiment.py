import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

# ---------------- Paths ----------------
DET_MODEL   = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL  = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH  = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"

# ---------------- Device ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"
print(f"[INFO] device={DEVICE}, half={USE_HALF}")

# ---------------- Detector ----------------
CONF_DET      = 0.4
IOU_DET       = 0.5
IMG_SIZE_DET  = 1280
PERSON_CLASS  = 0

# ---------------- Pose ------------------
POSE_INPUT_SIZE = 384
POSE_CONF_THR   = 0.01

# ---------------- Risk Threshold ----------------
TH_NORMAL = 0.45
TH_DANGER = 0.65
CLR_SAFE   = (0, 200, 0)
CLR_NORMAL = (0, 165, 255)
CLR_DANGER = (0, 0, 255)

# ---------------- Scoring Weights ----------------
W_PROX = 0.5
W_VEL  = 0.3
W_DIR  = 0.15
EMA_MU = 0.7

# ---------------- Velocity Normalization ----------------
VBOX_NORM  = 50.0
VFLOW_NORM = 2.0

# ---------------- Tracker Params ----------------
IOU_MATCH_THR      = 0.3
MAX_MISSES         = 15
ID_COUNTER_START   = 1
MIN_BOX_PIXELS     = 400

# ---------------- Utils ----------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / max(1e-6, area_a + area_b - inter)

def draw_skeleton(img, k):
    if k is None or k.size==0: return
    pairs=[(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),
           (5,6),(11,12),(5,11),(6,12),(0,5),(0,6)]
    for i,j in pairs:
        if i<k.shape[0] and j<k.shape[0]:
            p1=(int(k[i,0]),int(k[i,1]))
            p2=(int(k[j,0]),int(k[j,1]))
            cv2.line(img,p1,p2,(0,255,0),1)
    for i in range(k.shape[0]):
        cv2.circle(img,(int(k[i,0]),int(k[i,1])),2,(0,0,255),-1)

def to_numpy(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return x

def score_to_label(score):
    if score >= TH_DANGER:
        return "danger", CLR_DANGER
    elif score >= TH_NORMAL:
        return "normal", CLR_NORMAL
    else:
        return "safe", CLR_SAFE

def compute_proximity(all_by):
    all_by = np.asarray(all_by, dtype=np.float32)
    if all_by.size == 0: return np.array([])
    mn, mx = all_by.min(), all_by.max()
    return (all_by - mn) / max(1e-6, mx - mn)

# ---------------- Tracker Class -----------------
class Track:
    __slots__ = ("tid","box","misses","last_by","score_ema","hist_by","age")
    def __init__(self, tid, box):
        self.tid = tid
        self.box = list(map(int, box))
        self.misses = 0
        self.last_by = int(box[3])
        self.score_ema = None
        self.hist_by = deque(maxlen=5)
        self.hist_by.append(self.last_by)
        self.age = 0

def match_tracks(tracks, det_boxes, thr=IOU_MATCH_THR):
    if len(det_boxes)==0 or len(tracks)==0: return [-1]*len(det_boxes)
    used = set()
    matches = [-1]*len(det_boxes)
    for di, db in enumerate(det_boxes):
        best_iou, best_t = 0.0, -1
        for ti, tr in enumerate(tracks):
            if ti in used: continue
            iou = iou_xyxy(db, tr.box)
            if iou > best_iou:
                best_iou, best_t = iou, ti
        if best_iou >= thr:
            matches[di] = best_t
            used.add(best_t)
    return matches

# ---------------- GPU Optical Flow -----------------
class GpuOpticalFlow:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.have_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.nvof = None
        self.cuda_farneback = None
        self.cuda_tvl1 = None

        if self.have_cuda and hasattr(cv2.cuda, "NvidiaOpticalFlow_2_0"):
            try:
                nvof_cls = cv2.cuda.NvidiaOpticalFlow_2_0
                perfPreset = getattr(nvof_cls, "NVIDIA_OF_PERF_LEVEL_FAST", 0)
                self.nvof = nvof_cls.create(width, height, perfPreset=perfPreset)
                print("[INFO] Using NVIDIA HW Optical Flow 2.0")
            except Exception:
                print("[WARN] NVIDIA HW OF 속성 문제, Farneback/TV-L1 fallback")
        if self.nvof is None and self.have_cuda and hasattr(cv2.cuda, "FarnebackOpticalFlow"):
            self.cuda_farneback = cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=5, pyrScale=0.5, fastPyramids=False,
                winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0
            )
            print("[INFO] Using CUDA Farneback Optical Flow")
        if self.nvof is None and self.cuda_farneback is None and self.have_cuda and hasattr(cv2.cuda, "OpticalFlowDual_TVL1"):
            self.cuda_tvl1 = cv2.cuda.OpticalFlowDual_TVL1.create()
            print("[INFO] Using CUDA TV-L1 Optical Flow")
        if not any([self.nvof, self.cuda_farneback, self.cuda_tvl1]):
            raise RuntimeError("OpenCV CUDA Optical Flow가 활성화되어 있지 않습니다.")

        self.gpu_prev = cv2.cuda_GpuMat()
        self.gpu_curr = cv2.cuda_GpuMat()

    def compute(self, prev_gray_cpu, gray_cpu):
        self.gpu_prev.upload(prev_gray_cpu)
        self.gpu_curr.upload(gray_cpu)
        if self.nvof is not None:
            flow = self.nvof.calc(self.gpu_prev, self.gpu_curr, None)
            try:
                flow_f = self.nvof.convertToFloat(flow)
                return flow_f.download()
            except Exception:
                return flow.download()
        elif self.cuda_farneback is not None:
            return self.cuda_farneback.calc(self.gpu_prev, self.gpu_curr, None).download()
        else:
            return self.cuda_tvl1.calc(self.gpu_prev, self.gpu_curr, None).download()

def median_vflow_in_box(flow, box):
    if flow is None: return 0.0
    x1,y1,x2,y2 = list(map(int, box))
    H,W = flow.shape[:2]
    x1, x2 = max(0,min(W-1,x1)), max(0,min(W-1,x2))
    y1, y2 = max(0,min(H-1,y1)), max(0,min(H-1,y2))
    if x2<=x1 or y2<=y1: return 0.0
    vy = flow[y1:y2,x1:x2,1]
    if vy.size==0: return 0.0
    return float(np.median(vy))

# ---------------- Main -----------------
def main():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    det  = YOLO(DET_MODEL); pose = YOLO(POSE_MODEL)
    det.to(DEVICE); pose.to(DEVICE)
    try: det.fuse(); pose.fuse()
    except Exception: pass

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise SystemExit(f"cannot open {VIDEO_PATH}")

    ok, frame = cap.read()
    if not ok: raise SystemExit("no frames")
    H,W = frame.shape[:2]
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gpu_of = GpuOpticalFlow(W,H)

    tracks = []; next_tid = ID_COUNTER_START

    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = gpu_of.compute(prev_gray, gray)

        det_list = det.predict(
            frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET,
            classes=[PERSON_CLASS], verbose=False, device=DEVICE,
            half=USE_HALF
        )
        r0 = det_list[0]
        xyxy = to_numpy(r0.boxes.xyxy).astype(int) if len(r0.boxes) > 0 else np.array([])
        if xyxy.size > 0:
            areas = (xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1])
            xyxy = xyxy[areas>=MIN_BOX_PIXELS]
        all_by = xyxy[:,3] if xyxy.size>0 else np.array([])
        matches = match_tracks(tracks, xyxy.tolist()) if xyxy.size>0 else []
        prox_scores = compute_proximity(all_by) if all_by.size>0 else np.array([])

        used_tracks = set()
        for di, box in enumerate(xyxy):
            x1,y1,x2,y2 = map(int, box)
            by = int(y2)
            ti = matches[di] if di < len(matches) else -1
            if 0<=ti<len(tracks):
                tr = tracks[ti]
                dy_norm = max(0.0, min(1.0, (by-tr.last_by)/VBOX_NORM))
                vy_med = median_vflow_in_box(flow, box)
                vy_norm = max(0.0, min(1.0, vy_med/VFLOW_NORM))
                vel_score = 0.5*dy_norm + 0.5*vy_norm
                cx = (x1+x2)/2
                dir_factor = max(0.0, 1.0 - ((abs(cx-W/2)/(W/2))**2))
                raw_score = W_PROX*float(prox_scores[di]) + W_VEL*vel_score + W_DIR*dir_factor
                tr.box = [x1,y1,x2,y2]; tr.misses=0; tr.age+=1; tr.hist_by.append(by); tr.last_by=by
                tr.score_ema = raw_score if tr.score_ema is None else EMA_MU*tr.score_ema + (1-EMA_MU)*raw_score
                score_for_draw = tr.score_ema
                tid = tr.tid
                used_tracks.add(ti)
            else:
                tr = Track(next_tid,[x1,y1,x2,y2])
                tr.last_by=by
                vel_score=0.0
                dir_factor = max(0.0, 1.0 - ((x1+x2)/2-W/2)/(W/2)**2)
                raw_score = W_PROX*float(prox_scores[di]) + W_VEL*vel_score + W_DIR*dir_factor
                tr.score_ema=raw_score
                tracks.append(tr)
                tid = next_tid; next_tid+=1
                score_for_draw=raw_score

            crop = frame[y1:y2,x1:x2]
            if crop.size>0:
                pr_list = pose.predict(
                    crop, imgsz=POSE_INPUT_SIZE, conf=POSE_CONF_THR,
                    verbose=False, device=DEVICE, half=USE_HALF
                )
                if pr_list and pr_list[0].keypoints is not None:
                    kxy_all = to_numpy(pr_list[0].keypoints.xy)
                    if isinstance(kxy_all,np.ndarray) and kxy_all.ndim==3 and kxy_all.shape[0]>0:
                        best_iou,best_idx=0.0,-1
                        for i in range(kxy_all.shape[0]):
                            xs,ys=kxy_all[i,:,0],kxy_all[i,:,1]
                            bb=(xs.min(),ys.min(),xs.max(),ys.max())
                            giou=iou_xyxy((bb[0]+x1,bb[1]+y1,bb[2]+x1,bb[3]+y1),(x1,y1,x2,y2))
                            if giou>best_iou: best_iou,best_idx=giou,i
                        if best_idx>=0:
                            kxy = kxy_all[best_idx].astype(np.float32)
                            kxy[:,0]=np.clip(kxy[:,0]+x1,0,W-1)
                            kxy[:,1]=np.clip(kxy[:,1]+y1,0,H-1)
                            draw_skeleton(frame,kxy)

            risk_label,color = score_to_label(score_for_draw)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID {tid} | {risk_label} ({score_for_draw:.2f})",
                        (x1,max(15,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)

        for ti,tr in enumerate(tracks):
            if ti not in used_tracks: tr.misses+=1
        tracks=[t for t in tracks if t.misses<=MAX_MISSES]

        disp = cv2.resize(frame,(W//2,H//2))
        cv2.imshow("risk",disp)
        if cv2.waitKey(1) & 0xFF==ord('q'): break
        prev_gray=gray

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
