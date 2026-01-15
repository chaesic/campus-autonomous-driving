#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --------------- Paths & Configs ---------------
DET_MODEL = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"

IMG_SIZE_DET = 1280
IMG_SIZE_POSE = 384
CONF_DET = 0.4
IOU_DET = 0.5
PERSON_CLASS = 0
POSE_CONF_THR = 0.01

# ------------- IO / View ---------------
WINDOW_NAME = "a.py (BATCH - ALL FRAMES)"
VIEW_SCALE = 1.5

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
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {DEVICE}")

    det = YOLO(DET_MODEL)
    pose = YOLO(POSE_MODEL)
    print("[INFO] Models loaded")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {VIDEO_PATH}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(960 * VIEW_SCALE), int(540 * VIEW_SCALE))

    # FPS 측정
    t0 = time.perf_counter()
    frame_count = 0
    disp_color = (40, 230, 40)

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        det_results = det.predict(
            frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET,
            classes=[PERSON_CLASS], verbose=False, device=DEVICE
        )
        original_boxes = to_numpy(det_results[0].boxes.xyxy)

        if original_boxes is None or len(original_boxes) == 0:
            # FPS overlay
            frame_count += 1
            elapsed = max(1e-6, time.perf_counter() - t0)
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.2f}",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, disp_color, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break
            continue

        crops = []
        expanded_boxes_coords = []
        valid_original_boxes = []

        for box in original_boxes:
            box = box.astype(int)
            ex1, ey1, ex2, ey2 = expand(*box, W, H, scale=1.6)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size > 0:
                crops.append(crop)
                expanded_boxes_coords.append((ex1, ey1))
                valid_original_boxes.append(box)

        if crops:
            pose_results_batch = pose.predict(
                crops, imgsz=IMG_SIZE_POSE, conf=POSE_CONF_THR, verbose=False, device=DEVICE
            )
            for i, result in enumerate(pose_results_batch):
                if result.keypoints is None: continue
                original_box = valid_original_boxes[i]
                ex1, ey1 = expanded_boxes_coords[i]

                kxy_all = to_numpy(result.keypoints.xy)
                if not isinstance(kxy_all, np.ndarray) or kxy_all.ndim != 3 or kxy_all.shape[0] == 0: continue

                best_iou, best_idx = -1.0, -1
                for j in range(len(kxy_all)):
                    bb = kpts_to_bbox_xyxy(kxy_all[j])
                    if bb is None: continue
                    global_bb = (bb[0] + ex1, bb[1] + ey1, bb[2] + ex1, bb[3] + ey1)
                    iou_val = iou_xyxy(global_bb, original_box)
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, j

                if best_idx != -1:
                    kxy = kxy_all[best_idx].astype(np.float32)
                    kxy[:, 0] += ex1
                    kxy[:, 1] += ey1
                    draw_skeleton(frame, kxy)

        for box in valid_original_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

        # FPS overlay
        frame_count += 1
        elapsed = max(1e-6, time.perf_counter() - t0)
        fps = frame_count / elapsed
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, disp_color, 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()
    
    total_elapsed = time.perf_counter() - t0
    avg_fps = frame_count / max(1e-6, total_elapsed)
    print(f"[DONE] Total frames: {frame_count}, Elapsed: {total_elapsed:.2f}s, Avg FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
