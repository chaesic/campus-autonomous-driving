#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- Paths ----------------
DET_MODEL = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"

# ------------- Detector ----------------
CONF_DET = 0.4
IOU_DET = 0.5
IMG_SIZE_DET = 1280
PERSON_CLASS = 0

# --------------- Pose ------------------
POSE_INPUT_SIZE = 384
POSE_CONF_THR = 0.01

# ------------- IO / View ---------------
USE_IMSHOW = True
VIEW_SCALE = 1.2
WINDOW_NAME = "02_bbox_in_pose (ALL FRAMES)"

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
        (5,7),(7,9),(6,8),(8,10),
        (11,13),(13,15),(12,14),(14,16),
        (5,6),(11,12),(5,11),(6,12),
        (0,5),(0,6)
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
    print(f"[INFO] device={DEVICE}")

    det = YOLO(DET_MODEL)
    pose = YOLO(POSE_MODEL)
    print("[INFO] Models loaded")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO_PATH}")

    if USE_IMSHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, int(960*VIEW_SCALE), int(540*VIEW_SCALE))

    frame_idx = 0
    saved_with_pose = 0
    
    # FPS 측정
    t0 = time.perf_counter()
    frame_count = 0
    disp_color = (40, 230, 40)

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        # ---------- Detection ----------
        det_list = det.predict(
            frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET,
            classes=[PERSON_CLASS], verbose=False, device=DEVICE
        )
        r0 = det_list[0]
        
        if len(r0.boxes) == 0:
            pass
        else:
            xyxy = to_numpy(r0.boxes.xyxy).astype(int)
            conf = to_numpy(r0.boxes.conf)
            order = np.arange(len(xyxy)) if conf is None else np.argsort(-conf)

            # ---------- For each detection ----------
            for idx in order:
                x1,y1,x2,y2 = clamp(*xyxy[idx], W,H)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

                ex1,ey1,ex2,ey2 = expand(x1,y1,x2,y2, W,H, scale=1.6)
                crop = frame[ey1:ey2, ex1:ex2]
                if crop.size == 0:
                    continue

                # ---------- Pose on crop ----------
                pr_list = pose.predict(
                    crop, imgsz=POSE_INPUT_SIZE, conf=POSE_CONF_THR, verbose=False, device=DEVICE
                )
                if not pr_list:
                    continue
                pr = pr_list[0]
                if pr.keypoints is None:
                    continue

                kxy_all = to_numpy(pr.keypoints.xy)  # (N,K,2)
                if (not isinstance(kxy_all, np.ndarray)) or kxy_all.ndim != 3:
                    continue
                N = int(kxy_all.shape[0])
                if N <= 0 or kxy_all.shape[1] < 4 or kxy_all.shape[2] != 2:
                    continue

                # detector 박스와 IoU 최대 인스턴스 선택
                best_iou, best_idx = 0.0, -1
                for i in range(N):
                    bb = kpts_to_bbox_xyxy(kxy_all[i])
                    if bb is None: continue
                    giou = iou_xyxy((bb[0]+ex1, bb[1]+ey1, bb[2]+ex1, bb[3]+ey1), (x1,y1,x2,y2))
                    if giou > best_iou:
                        best_iou, best_idx = giou, i
                if best_idx < 0:
                    continue

                kxy = kxy_all[best_idx].astype(np.float32)  # (K,2)
                kxy[:,0] = np.clip(kxy[:,0] + ex1, 0, W-1)
                kxy[:,1] = np.clip(kxy[:,1] + ey1, 0, H-1)
                draw_skeleton(frame, kxy)
                saved_with_pose += 1

        # ---------- FPS overlay ----------
        frame_count += 1
        elapsed = max(1e-6, time.perf_counter() - t0)
        fps = frame_count / elapsed
        cv2.putText(frame, f"FPS: {fps:.2f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, disp_color, 2, cv2.LINE_AA)

        # ---------- Live View ----------
        if USE_IMSHOW:
            if VIEW_SCALE != 1.0:
                view = cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE, interpolation=cv2.INTER_LINEAR)
            else:
                view = frame
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        frame_idx += 1

    cap.release()
    if USE_IMSHOW:
        cv2.destroyAllWindows()
    
    total_elapsed = time.perf_counter() - t0
    avg_fps = frame_count / max(1e-6, total_elapsed)
    print(f"[DONE] Total frames: {frame_count}, Elapsed: {total_elapsed:.2f}s, Avg FPS: {avg_fps:.2f}")
    print(f"[DONE] frames_with_pose={saved_with_pose}")

if __name__ == "__main__":
    main()
