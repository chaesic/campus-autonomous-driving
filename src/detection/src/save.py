#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import imageio

DET_MODEL   = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL  = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH  = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"
OUTPUT_PATH = "output_openpose_v4.mp4" 

CONF_DET      = 0.4
IOU_DET       = 0.5
IMG_SIZE_DET  = 1280
PERSON_CLASS  = 0

POSE_INPUT_SIZE = 384
POSE_CONF_THR   = 0.01

USE_IMSHOW  = True
VIEW_SCALE  = 1.2
WINDOW_NAME = "test"

BOX_COLOR = (0, 255, 0) 
BOX_THICK = 2         

# (0, 1) 코-왼눈, (0, 2) 코-오른눈, (1, 3) 왼눈-왼귀, (2, 4) 오른눈-오른귀
# (5, 6) 왼어깨-오른어깨, (11, 12) 왼엉덩이-오른엉덩이
# (5, 11) 왼어깨-왼엉덩이, (6, 12) 오른어깨-오른엉덩이
# (5, 7) 왼어깨-왼팔꿈치, (7, 9) 왼팔꿈치-왼손목
# (6, 8) 오른어깨-오른팔꿈치, (8, 10) 오른팔꿈치-오른손목
# (11, 13) 왼엉덩이-왼무릎, (13, 15) 왼무릎-왼발목
# (12, 14) 오른엉덩이-오른무릎, (14, 16) 오른무릎-오른발목

LIMB_COLORS_MAP = [

    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), # (코-눈, 눈-귀)

    # 몸통 (어깨-어깨, 엉덩이-엉덩이): 빨간색
    (0, 0, 255), (0, 0, 255), # (왼어깨-오른어깨), (왼엉덩이-오른엉덩이)

    # 몸통 라인 (어깨-엉덩이): 빨간색
    (0, 0, 255), (0, 0, 255), # (왼어깨-왼엉덩이), (오른어깨-오른엉덩이)

    # 왼팔 (사람 기준 왼쪽 팔, 이미지 기준 오른쪽 팔): 초록색 계열
    (0, 255, 85), (0, 170, 0), # (왼어깨-왼팔꿈치: 연두), (왼팔꿈치-왼손목: 진한 초록)

    # 오른팔 (사람 기준 오른쪽 팔, 이미지 기준 왼쪽 팔): 주황색-노란색 계열
    (0, 170, 255), (0, 85, 255), # (오른어깨-오른팔꿈치: 주황), (오른팔꿈치-오른손목: 노랑)

    # 왼다리 (사람 기준 왼쪽 다리, 이미지 기준 오른쪽 다리): 파란색 계열
    (255, 170, 0), (255, 85, 0), # (왼엉덩이-왼무릎: 하늘색), (왼무릎-왼발목: 파란색)

    # 오른다리 (사람 기준 오른쪽 다리, 이미지 기준 왼쪽 다리): 보라색 계열
    (170, 0, 255), (85, 0, 255)  # (오른엉덩이-오른무릎: 보라), (오른무릎-오른발목: 진한 보라)
]

SKEL_THICK = 2   

KPT_COLORS_MAP = [
    # 얼굴 (코, 눈, 귀): 노랑-초록 계열
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), # 코, 왼눈, 오른눈, 왼귀, 오른귀

    # 팔 (어깨, 팔꿈치, 손목)
    (0, 255, 0), (0, 255, 0), # 왼어깨, 오른어깨 (초록)
    (0, 255, 85), (0, 170, 255), # 왼팔꿈치 (연두), 오른팔꿈치 (주황)
    (0, 170, 0), (0, 85, 255), # 왼손목 (진한 초록), 오른손목 (노랑)

    # 다리 (엉덩이, 무릎, 발목)
    (0, 255, 0), (0, 255, 0), # 왼엉덩이, 오른엉덩이 (초록)
    (255, 170, 0), (170, 0, 255), # 왼무릎 (하늘), 오른무릎 (보라)
    (255, 85, 0), (85, 0, 255)  # 왼발목 (파랑), 오른발목 (진한 보라)
]
KPT_RADIUS = 1              # Keypoint 점 반지름

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

def to_numpy(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return x

def draw_skeleton(img, k, thick_sk, radius_kp):
    pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (11, 12),
        (5, 11), (6, 12),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    K = int(k.shape[0])
    
    for i, pair in enumerate(pairs):
        p1_idx, p2_idx = pair
        if 0 <= p1_idx < K and 0 <= p2_idx < K:
            p1 = (int(k[p1_idx,0]), int(k[p1_idx,1]))
            p2 = (int(k[p2_idx,0]), int(k[p2_idx,1]))
            color = LIMB_COLORS_MAP[i % len(LIMB_COLORS_MAP)]
            cv2.line(img, p1, p2, color, thick_sk, cv2.LINE_AA)
    
    for i in range(K):
        if 0 <= i < len(KPT_COLORS_MAP):
            kpt_color = KPT_COLORS_MAP[i]
        else:
            kpt_color = (255,255,255)
        cv2.circle(img, (int(k[i,0]), int(k[i,1])), radius_kp, kpt_color, -1, cv2.LINE_AA)

def main():
    det  = YOLO(DET_MODEL)
    pose = YOLO(POSE_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = imageio.get_writer(OUTPUT_PATH, fps=fps, codec='libx264', macro_block_size=1)

    if USE_IMSHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, int(960*VIEW_SCALE), int(540*VIEW_SCALE))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        det_list = det.predict(
            frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET,
            classes=[PERSON_CLASS], verbose=False
        )
        r0 = det_list[0]

        if len(r0.boxes) > 0:
            xyxy = to_numpy(r0.boxes.xyxy).astype(int)
            conf = to_numpy(r0.boxes.conf)
            order = np.argsort(-conf)

            for idx in order:
                x1,y1,x2,y2 = clamp(*xyxy[idx], W,H)
                cv2.rectangle(frame, (x1,y1), (x2,y2), BOX_COLOR, BOX_THICK)

                ex1,ey1,ex2,ey2 = expand(x1,y1,x2,y2, W,H, scale=1.6)
                crop = frame[ey1:ey2, ex1:ex2]
                if crop.size == 0:
                    continue

                pr_list = pose.predict(
                    crop, imgsz=POSE_INPUT_SIZE, conf=POSE_CONF_THR, verbose=False
                )
                if not pr_list or pr_list[0].keypoints is None:
                    continue

                kxy_all = to_numpy(pr_list[0].keypoints.xy)
                if not isinstance(kxy_all, np.ndarray) or kxy_all.ndim != 3 or kxy_all.shape[0] == 0:
                    continue

                best_iou, best_idx = 0.0, -1
                for i in range(kxy_all.shape[0]):
                    bb = kpts_to_bbox_xyxy(kxy_all[i])
                    if bb is None: continue
                    giou = iou_xyxy((bb[0]+ex1, bb[1]+ey1, bb[2]+ex1, bb[3]+ey1), (x1,y1,x2,y2))
                    if giou > best_iou:
                        best_iou, best_idx = giou, i
                if best_idx < 0:
                    continue

                kxy = kxy_all[best_idx].astype(np.float32)
                kxy[:,0] = np.clip(kxy[:,0] + ex1, 0, W-1)
                kxy[:,1] = np.clip(kxy[:,1] + ey1, 0, H-1)
                
                draw_skeleton(frame, kxy, SKEL_THICK, KPT_RADIUS)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

        if USE_IMSHOW:
            view = cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE) if VIEW_SCALE != 1.0 else frame
            cv2.imshow(WINDOW_NAME, view)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        frame_idx += 1
        
    writer.close()
    cap.release()
    if USE_IMSHOW:
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()