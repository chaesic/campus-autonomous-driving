import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import imageio  # For video saving

# --------------- Paths & Configs ---------------
DET_MODEL   = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL  = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH  = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"
OUTPUT_PATH = "output.mp4" 

IMG_SIZE_DET  = 1280
IMG_SIZE_POSE = 384
CONF_DET      = 0.4
IOU_DET       = 0.5
PERSON_CLASS  = 0
POSE_CONF_THR   = 0.01

# ------------- IO / View ---------------
WINDOW_NAME = "YOLO Pose"
VIEW_SCALE  = 1.5
USE_IMSHOW  = True

# --------------- Utils -----------------
def clamp(x1,y1,x2,y2,W,H):
    x1 = int(max(0,min(W-1,x1))); y1=int(max(0,min(H-1,y1)))
    x2 = int(max(0,min(W-1,x2))); y2=int(max(0,min(H-1,y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1,y1,x2,y2

def expand(x1,y1,x2,y2,W,H,scale=1.6):
    w = x2 - x1; h = y2 - y1
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    nw = max(2.0, w * scale); nh = max(2.0, h * scale)
    ex1 = int(max(0, cx - nw/2.0)); ex2 = int(min(W-1, cx + nw/2.0))
    ey1 = int(max(0, cy - nh/2.0)); ey2 = int(min(H-1, cy + nh/2.0))
    return ex1,ey1,ex2,ey2

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / max(1e-6, area_a + area_b - inter)

def kpts_to_bbox_xyxy(k_xy):
    if k_xy is None or len(k_xy)==0: return None
    xs = k_xy[:,0]; ys = k_xy[:,1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(xs.max())

def draw_skeleton(img, k):
    pairs = [
        (5,7),(7,9),(6,8),(8,10), (11,13),(13,15),(12,14),(14,16),
        (5,6),(11,12),(5,11),(6,12), (0,5),(0,6)
    ]
    K = int(k.shape[0])
    for i, j in pairs:
        if 0 <= i < K and 0 <= j < K:
            p1 = (int(k[i,0]), int(k[i,1]))
            p2 = (int(k[j,0]), int(k[j,1]))
            cv2.line(img, p1, p2, (0,255,0), 2)
    for i in range(K):
        cv2.circle(img, (int(k[i,0]), int(k[i,1])), 3, (0,0,255), -1)

def to_numpy(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return x

# --------------- Main ------------------
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {DEVICE}")

    det = YOLO(DET_MODEL)
    pose = YOLO(POSE_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {VIDEO_PATH}")

    # Get video properties for writer ---
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video properties: {W}x{H} @ {fps:.2f} FPS")

    # Setup imageio video writer ---
    writer = imageio.get_writer(OUTPUT_PATH, fps=fps, codec='libx264', macro_block_size=1)

    if USE_IMSHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, int(960 * VIEW_SCALE), int(540 * VIEW_SCALE))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        det_results = det.predict(frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET, classes=[PERSON_CLASS], verbose=False, device=DEVICE)
        
        r0 = det_results[0]
        if len(r0.boxes) > 0:
            original_boxes = to_numpy(r0.boxes.xyxy)
            conf = to_numpy(r0.boxes.conf)
            order = np.argsort(-conf)

            for idx in order:
                box = original_boxes[idx].astype(int)
                x1,y1,x2,y2 = clamp(*box, W,H)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

                ex1, ey1, ex2, ey2 = expand(x1,y1,x2,y2, W, H, scale=1.6)
                crop = frame[ey1:ey2, ex1:ex2]

                if crop.size > 0:
                    pose_results = pose.predict(crop, imgsz=IMG_SIZE_POSE, conf=POSE_CONF_THR, verbose=False, device=DEVICE)
                    
                    pr = pose_results[0]
                    if pr.keypoints is not None:
                        kxy_all = to_numpy(pr.keypoints.xy)
                        if isinstance(kxy_all, np.ndarray) and kxy_all.ndim == 3 and kxy_all.shape[0] > 0:
                            best_iou, best_idx = -1.0, -1
                            for j in range(len(kxy_all)):
                                bb = kpts_to_bbox_xyxy(kxy_all[j])
                                if bb is None: continue
                                global_bb = (bb[0] + ex1, bb[1] + ey1, bb[2] + ex1, bb[3] + ey1)
                                iou_val = iou_xyxy(global_bb, box)
                                if iou_val > best_iou:
                                    best_iou, best_idx = iou_val, j
                            
                            if best_idx != -1:
                                kxy = kxy_all[best_idx].astype(np.float32)
                                kxy[:, 0] += ex1
                                kxy[:, 1] += ey1
                                draw_skeleton(frame, kxy)

        # Save the final frame to video ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

        if USE_IMSHOW:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

        frame_idx += 1

    # Finalize video saving ---
    writer.close()
    cap.release()
    if USE_IMSHOW:
        cv2.destroyAllWindows()
    
    print(f"\n[INFO] Done. Processed {frame_idx} frames.")
    print(f"Video saved successfully to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()