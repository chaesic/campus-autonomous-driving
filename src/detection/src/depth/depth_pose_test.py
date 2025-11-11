import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

DET_MODEL_PATH   = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL_PATH  = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
DEPTH_MODEL_DIR  = "/home/jaemin/yolopose/model/"
VIDEO_PATH       = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE_DET  = 1280
IMG_SIZE_POSE = 384
CONF_DET      = 0.4
IOU_DET       = 0.5
PERSON_CLASS  = 0
POSE_CONF_THR = 0.01
DEPTH_ENCODER = 'vits' 
CROP_BOTTOM_PERCENT = 20
CLIP_BOTTOM_PERCENTILE = 5

# 이 값보다 작은 면적(width*height)을 가진 객체는 Pose 추정에서 제외
MIN_DETECTION_AREA = 3000 

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
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

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
    print(f"[INFO] Using device: {DEVICE}")

    det_model = YOLO(DET_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    model_configs = { 'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}, }
    depth_model = DepthAnythingV2(**model_configs[DEPTH_ENCODER])
    depth_model_path = os.path.expanduser(f'{DEPTH_MODEL_DIR}/depth_anything_v2_{DEPTH_ENCODER}.pth')
    depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    print("[INFO] All models loaded successfully.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {VIDEO_PATH}")

    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    display_w, display_h = 960, 540
    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Estimation", display_w, display_h)
    cv2.resizeWindow("Depth Video", display_w, display_h)
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        det_results = det_model.predict(frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET, classes=[PERSON_CLASS], verbose=False, device=DEVICE)
        original_boxes = to_numpy(det_results[0].boxes.xyxy)
        pose_frame = frame.copy()
        
        if original_boxes is not None and len(original_boxes) > 0:
            crops, expanded_coords, valid_boxes = [], [], []
            for box in original_boxes:
                box = box.astype(int)
                
                # BBox 면적 계산 및 필터링
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                if box_w * box_h < MIN_DETECTION_AREA:
                    continue # 면적이 작으면 건너뛰기

                ex1, ey1, ex2, ey2 = expand(*box, orig_width, orig_height)
                crop = frame[ey1:ey2, ex1:ex2]
                if crop.size > 0:
                    crops.append(crop)
                    expanded_coords.append((ex1, ey1))
                    valid_boxes.append(box)

            if crops:
                pose_results = pose_model.predict(crops, imgsz=IMG_SIZE_POSE, conf=POSE_CONF_THR, verbose=False, device=DEVICE)
                for i, res in enumerate(pose_results):
                    if res.keypoints is None: continue
                    kxy_all = to_numpy(res.keypoints.xy)
                    if not isinstance(kxy_all, np.ndarray) or kxy_all.ndim != 3 or kxy_all.shape[0] == 0: continue
                    best_iou, best_idx = -1.0, -1
                    for j in range(len(kxy_all)):
                        bb = kpts_to_bbox_xyxy(kxy_all[j])
                        if bb is None: continue
                        global_bb = (bb[0] + expanded_coords[i][0], bb[1] + expanded_coords[i][1], bb[2] + expanded_coords[i][0], bb[3] + expanded_coords[i][1])
                        iou_val = iou_xyxy(global_bb, valid_boxes[i])
                        if iou_val > best_iou: best_iou, best_idx = iou_val, j
                    if best_idx != -1:
                        kxy = kxy_all[best_idx].astype(np.float32)
                        kxy[:, 0] += expanded_coords[i][0]
                        kxy[:, 1] += expanded_coords[i][1]
                        draw_skeleton(pose_frame, kxy)

            # 필터링을 통과한 valid_boxes만 화면에 그림
            for box in valid_boxes:
                cv2.rectangle(pose_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        
        crop_pixels = int(orig_height * CROP_BOTTOM_PERCENT / 100)
        frame_cropped = frame[:orig_height - crop_pixels, :]
        inference_dim = (int(orig_width / 2), int((orig_height - crop_pixels) / 2))
        frame_for_inference = cv2.resize(frame_cropped, inference_dim, interpolation=cv2.INTER_AREA)
        depth = depth_model.infer_image(frame_for_inference)
        vmin = np.percentile(depth, CLIP_BOTTOM_PERCENTILE)
        depth_clipped = np.clip(depth, vmin, np.max(depth))
        depth_normalized = cv2.normalize(depth_clipped, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        pose_frame_display = cv2.resize(pose_frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
        target_depth_h = int(display_h * (orig_height - crop_pixels) / orig_height)
        depth_colored_display = cv2.resize(depth_colored, (display_w, target_depth_h), interpolation=cv2.INTER_AREA)
            
        cv2.imshow("Pose Estimation", pose_frame_display)
        cv2.imshow("Depth Video", depth_colored_display)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()