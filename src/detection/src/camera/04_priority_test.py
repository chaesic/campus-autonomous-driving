#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import csv
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
from scipy import stats
from datetime import datetime

# ==================== CONFIGURATION ====================
DET_MODEL = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"

CONF_DET = 0.4
IOU_DET = 0.5
IMG_SIZE_DET = 1280
PERSON_CLASS = 0

POSE_INPUT_SIZE = 384
POSE_CONF_THR = 0.01

WINDOW_NAME = "Priority View"
VIEW_SCALE = 1.2

# ========== Kalman Filter (Position) ==========
KALMAN_PROCESS_NOISE = 0.01
KALMAN_MEASUREMENT_NOISE = 1.0
KALMAN_Q = np.array([[KALMAN_PROCESS_NOISE, 0, 0, 0],
                      [0, KALMAN_PROCESS_NOISE, 0, 0],
                      [0, 0, KALMAN_PROCESS_NOISE, 0],
                      [0, 0, 0, KALMAN_PROCESS_NOISE]])
KALMAN_R = np.array([[KALMAN_MEASUREMENT_NOISE, 0],
                      [0, KALMAN_MEASUREMENT_NOISE]])

# ========== Distance Kalman Filter ==========
DISTANCE_KALMAN_PROCESS_NOISE = 0.08
DISTANCE_KALMAN_MEASUREMENT_NOISE = 0.3
DISTANCE_KALMAN_WARMUP = 1

# ========== History & Trend ==========
HISTORY_LENGTH = 15
TREND_WINDOW = 8

# ========== Distance Thresholds ==========
DISTANCE_THRESHOLDS = {
    "SAFE": 12.0,
    "NORMAL": 6.0,
    "DANGER": 2.5,
}

# ========== Trajectory ==========
TRAJECTORY_CENTER_REGION = 0.35
TRAJECTORY_HIGH_PRIORITY_THRESHOLD = 0.3

# ========== Hysteresis ==========
HYSTERESIS = 0.10

# Safe Detection Limit
MAX_SAFE_DETECTIONS = 5

# Camera Model
FOCAL_LENGTH_PIXELS = 800
PERSON_HEIGHT_METERS = 1.7

# Vehicle
FPS = 30.0
VEHICLE_SPEED_MPS = 6.7

# ========== 디버깅: 로깅 활성화 ==========
ENABLE_DEBUG_LOGGING = True
DEBUG_LOG_DIR = "./debug_logs"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

# ==================== Distance Kalman Filter =================
class DistanceKalmanFilter:
    def __init__(self, initial_distance):
        self.x = initial_distance
        self.P = 1.0
        self.Q = DISTANCE_KALMAN_PROCESS_NOISE
        self.R = DISTANCE_KALMAN_MEASUREMENT_NOISE
        self.frame_count = 0
    
    def filter(self, z_distance):
        if self.frame_count < DISTANCE_KALMAN_WARMUP:
            self.x = z_distance
            self.frame_count += 1
            return self.x
        
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z_distance - self.x)
        self.P = (1.0 - K) * self.P
        
        self.frame_count += 1
        return self.x
    
    def get_distance(self):
        return self.x

# ==================== Position Kalman Filter =================
class SimpleKalmanFilter:
    def __init__(self, x0, y0, dt=0.033):
        self.dt = dt
        self.x = np.array([x0, y0, 0.0, 0.0])
        self.P = np.eye(4) * 10.0
        
        self.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float32)
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + KALMAN_Q
        return self.x[:2]
    
    def update(self, z):
        z = np.array(z, dtype=np.float32)
        S = self.H @ self.P @ self.H.T + KALMAN_R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]
    
    def get_position(self):
        return self.x[:2]

# ==================== Utils =================
def clamp(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, x1))); y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2))); y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def expand(x1, y1, x2, y2, W, H, scale=1.6):
    w = x2 - x1; h = y2 - y1
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    nw = max(2.0, w * scale); nh = max(2.0, h * scale)
    ex1 = int(max(0, cx - nw / 2.0)); ex2 = int(min(W - 1, cx + nw / 2.0))
    ey1 = int(max(0, cy - nh / 2.0)); ey2 = int(min(H - 1, cy + nh / 2.0))
    return ex1, ey1, ex2, ey2

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)

def kpts_to_bbox_xyxy(k_xy):
    if k_xy is None or len(k_xy) == 0: return None
    xs = k_xy[:, 0]; ys = k_xy[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def draw_skeleton(img, k):
    pairs = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12),
        (0, 5), (0, 6)
    ]
    for i, j in pairs:
        if 0 <= i < k.shape[0] and 0 <= j < k.shape[0]:
            p1 = (int(k[i, 0]), int(k[i, 1]))
            p2 = (int(k[j, 0]), int(k[j, 1]))
            cv2.line(img, p1, p2, (0, 255, 0), 2)
    for i in range(k.shape[0]):
        cv2.circle(img, (int(k[i, 0]), int(k[i, 1])), 3, (0, 0, 255), -1)

def to_numpy(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return x

# ==================== Risk Assessment =================

def estimate_distance_from_bbox(bbox, image_height=720):
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    
    if bbox_height < 10:
        return float('inf')
    
    focal_length_px = FOCAL_LENGTH_PIXELS * (image_height / 720.0)
    distance_m = (focal_length_px * PERSON_HEIGHT_METERS) / bbox_height
    
    return max(0.5, distance_m)

def calculate_trajectory_risk(bbox, W, H):
    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    vehicle_cx = W / 2.0
    
    normalized_dist = abs(bbox_cx - vehicle_cx) / (W / 2.0)
    
    if normalized_dist <= TRAJECTORY_CENTER_REGION:
        trajectory_risk = 1.0
    else:
        trajectory_risk = max(0.0, 1.0 - (normalized_dist - TRAJECTORY_CENTER_REGION) 
                             / (1.0 - TRAJECTORY_CENTER_REGION))
    
    return trajectory_risk

def calculate_distance_trend(history_dict):
    if len(history_dict["filtered_distances"]) < TREND_WINDOW:
        return 0.0, False
    
    recent_distances = history_dict["filtered_distances"][-TREND_WINDOW:]
    frame_indices = np.arange(len(recent_distances))
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(frame_indices, recent_distances)
        is_approaching = slope < -0.08
        return slope, is_approaching
    except:
        return 0.0, False

def assess_risk_v5_debug(track_id, bbox, distance_kf, history_dict, W, H, prev_state=None):
    """
    위험도 평가 + 디버깅 정보 반환
    """
    x1, y1, x2, y2 = bbox
    
    # 거리 계산
    raw_distance_m = estimate_distance_from_bbox(bbox)
    filtered_distance_m = distance_kf.filter(raw_distance_m)
    
    history_dict["raw_distances"].append(raw_distance_m)
    history_dict["filtered_distances"].append(filtered_distance_m)
    
    if len(history_dict["raw_distances"]) > HISTORY_LENGTH:
        history_dict["raw_distances"].pop(0)
        history_dict["filtered_distances"].pop(0)
    
    distance = filtered_distance_m
    bbox_height = y2 - y1
    
    # 위험도 계산
    if distance >= DISTANCE_THRESHOLDS["SAFE"]:
        base_risk = 0.0
    elif distance >= DISTANCE_THRESHOLDS["NORMAL"]:
        base_risk = 0.30 * (1.0 - (distance - DISTANCE_THRESHOLDS["NORMAL"]) 
                           / (DISTANCE_THRESHOLDS["SAFE"] - DISTANCE_THRESHOLDS["NORMAL"]))
    elif distance >= DISTANCE_THRESHOLDS["DANGER"]:
        base_risk = 0.30 + 0.25 * (1.0 - (distance - DISTANCE_THRESHOLDS["DANGER"]) 
                                   / (DISTANCE_THRESHOLDS["NORMAL"] - DISTANCE_THRESHOLDS["DANGER"]))
    else:
        base_risk = 0.55 + 0.45 * (1.0 - distance / DISTANCE_THRESHOLDS["DANGER"])
    
    risk_score = base_risk
    risk_score = np.clip(risk_score, 0.0, 1.0)
    
    # 상태 판정
    if prev_state is None:
        if risk_score < 0.30:
            state = "SAFE"
            color = (0, 255, 0)
        elif risk_score < 0.55:
            state = "NORMAL"
            color = (0, 165, 255)
        else:
            state = "DANGER"
            color = (0, 0, 255)
        hysteresis_applied = False
    else:
        hysteresis_applied = False
        if prev_state == "SAFE":
            if risk_score > 0.30 + HYSTERESIS:
                state = "NORMAL"
                color = (0, 165, 255)
                hysteresis_applied = True
            else:
                state = "SAFE"
                color = (0, 255, 0)
        elif prev_state == "NORMAL":
            if risk_score < 0.30 - HYSTERESIS:
                state = "SAFE"
                color = (0, 255, 0)
                hysteresis_applied = True
            elif risk_score > 0.55 - HYSTERESIS:
                state = "DANGER"
                color = (0, 0, 255)
                hysteresis_applied = True
            else:
                state = "NORMAL"
                color = (0, 165, 255)
        else:  # DANGER
            if risk_score < 0.55 - HYSTERESIS:
                state = "NORMAL"
                color = (0, 165, 255)
                hysteresis_applied = True
            else:
                state = "DANGER"
                color = (0, 0, 255)
    
    # 우선순위
    trajectory_risk = calculate_trajectory_risk(bbox, W, H)
    if trajectory_risk > TRAJECTORY_HIGH_PRIORITY_THRESHOLD:
        is_high_priority = (state == "NORMAL" or state == "DANGER")
    else:
        is_high_priority = (state == "DANGER")
    
    # ===== 디버깅 정보 =====
    debug_info = {
        "track_id": track_id,
        "bbox_height": bbox_height,
        "raw_distance_m": raw_distance_m,
        "filtered_distance_m": filtered_distance_m,
        "distance_kf_frame_count": distance_kf.frame_count,
        "base_risk": base_risk,
        "final_risk": risk_score,
        "prev_state": prev_state if prev_state else "None",
        "new_state": state,
        "hysteresis_applied": hysteresis_applied,
        "trajectory_risk": trajectory_risk,
        "is_high_priority": is_high_priority,
    }
    
    return risk_score, state, color, is_high_priority, distance, debug_info

# ==================== Main =================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={DEVICE}")

    det = YOLO(DET_MODEL)
    pose = YOLO(POSE_MODEL)
    print("[INFO] Models loaded")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO_PATH}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(960 * VIEW_SCALE), int(540 * VIEW_SCALE))

    # ===== 디버깅: CSV 파일 준비 =====
    if ENABLE_DEBUG_LOGGING:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(DEBUG_LOG_DIR, f"risk_debug_{timestamp}.csv")
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            'frame_id', 'track_id', 'bbox_height', 'raw_distance_m', 'filtered_distance_m',
            'distance_kf_frame_count', 'base_risk', 'final_risk', 'prev_state', 'new_state',
            'hysteresis_applied', 'trajectory_risk', 'is_high_priority'
        ])
        csv_writer.writeheader()
        print(f"[INFO] Debug log file: {csv_filename}")
    
    kalman_filters = {}
    distance_kf_filters = {}
    track_histories = {}
    track_states = {}
    next_track_id = 0
    prev_detections = []
    track_id_map = {}

    t0 = time.perf_counter()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        # 검출
        det_list = det.predict(
            frame, imgsz=IMG_SIZE_DET, conf=CONF_DET, iou=IOU_DET,
            classes=[PERSON_CLASS], verbose=False, device=DEVICE
        )
        r0 = det_list[0]

        current_detections_with_scores = []
        
        if len(r0.boxes) > 0:
            xyxy = to_numpy(r0.boxes.xyxy).astype(int)
            conf = to_numpy(r0.boxes.conf)
            order = np.argsort(-conf)

            for idx in order:
                x1, y1, x2, y2 = clamp(*xyxy[idx], W, H)
                current_detections_with_scores.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf[idx],
                    "det_idx": idx
                })

        # 트래킹
        current_bboxes = [d["bbox"] for d in current_detections_with_scores]
        
        matched_pairs = []
        used_current = set()
        for prev_idx, prev_bbox in enumerate(prev_detections):
            best_iou = 0.0
            best_curr_idx = -1
            for curr_idx, curr_bbox in enumerate(current_bboxes):
                if curr_idx in used_current:
                    continue
                giou = iou_xyxy(prev_bbox, curr_bbox)
                if giou > best_iou and giou > 0.3:
                    best_iou = giou
                    best_curr_idx = curr_idx
            if best_curr_idx >= 0:
                matched_pairs.append((prev_idx, best_curr_idx))
                used_current.add(best_curr_idx)

        new_track_id_map = {}
        for prev_idx, curr_idx in matched_pairs:
            if prev_idx in track_id_map:
                track_id = track_id_map[prev_idx]
            else:
                track_id = next_track_id
                next_track_id += 1
            new_track_id_map[curr_idx] = track_id

        for curr_idx in range(len(current_detections_with_scores)):
            if curr_idx not in new_track_id_map:
                new_track_id_map[curr_idx] = next_track_id
                next_track_id += 1

        track_id_map = new_track_id_map

        # ===== 위험도 평가 =====
        all_risks = {}
        
        for det_idx, det_data in enumerate(current_detections_with_scores):
            x1, y1, x2, y2 = det_data["bbox"]
            track_id = track_id_map.get(det_idx, -1)
            bbox_cx = (x1 + x2) / 2.0
            bbox_cy = (y1 + y2) / 2.0

            if track_id not in kalman_filters:
                kalman_filters[track_id] = SimpleKalmanFilter(bbox_cx, bbox_cy)
                distance_kf_filters[track_id] = DistanceKalmanFilter(estimate_distance_from_bbox((x1, y1, x2, y2)))
                track_histories[track_id] = {"raw_distances": [], "filtered_distances": [], "states": []}
                track_states[track_id] = None
            
            kf = kalman_filters[track_id]
            dist_kf = distance_kf_filters[track_id]
            hist = track_histories[track_id]
            
            kf.predict()
            kf.update([bbox_cx, bbox_cy])

            prev_state = track_states[track_id]
            risk_score, state, color, is_high_priority, distance_m, debug_info = assess_risk_v5_debug(
                track_id, (x1, y1, x2, y2), dist_kf, hist, W, H, prev_state
            )
            track_states[track_id] = state
            hist["states"].append(state)
            
            all_risks[det_idx] = (risk_score, state, color, is_high_priority, distance_m)
            
            # ===== 디버깅: CSV 저장 =====
            if ENABLE_DEBUG_LOGGING:
                debug_info['frame_id'] = frame_count
                csv_writer.writerow(debug_info)

        # ===== SAFE 필터링 =====
        safe_indices = [idx for idx, (_, state, _, _, _) in all_risks.items() if state == "SAFE"]
        safe_by_conf = sorted(
            [(idx, current_detections_with_scores[idx]["conf"]) for idx in safe_indices],
            key=lambda x: x[1],
            reverse=True
        )[:MAX_SAFE_DETECTIONS]
        safe_keep_set = set(idx for idx, _ in safe_by_conf)

        # 시각화
        high_priority_peds = []

        for det_idx, det_data in enumerate(current_detections_with_scores):
            if det_idx not in all_risks:
                continue
            
            risk_score, state, color, is_high_priority, distance_m = all_risks[det_idx]
            x1, y1, x2, y2 = det_data["bbox"]
            
            if state == "SAFE" and det_idx not in safe_keep_set:
                continue
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            display_text = f"{state} ({risk_score:.2f}) {distance_m:.1f}m"
            cv2.putText(frame, display_text,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            if state == "SAFE":
                continue

            if is_high_priority:
                high_priority_peds.append({
                    "track_id": track_id_map.get(det_idx, -1),
                    "bbox": (x1, y1, x2, y2),
                    "risk_score": risk_score,
                    "state": state,
                    "distance_m": distance_m
                })

            ex1, ey1, ex2, ey2 = expand(x1, y1, x2, y2, W, H, scale=1.6)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0: continue

            pr_list = pose.predict(
                crop, imgsz=POSE_INPUT_SIZE, conf=POSE_CONF_THR, verbose=False, device=DEVICE
            )
            if not pr_list: continue
            pr = pr_list[0]
            if pr.keypoints is None: continue

            kxy_all = to_numpy(pr.keypoints.xy)
            if not isinstance(kxy_all, np.ndarray) or kxy_all.ndim != 3: continue
            N = kxy_all.shape[0]
            if N <= 0: continue

            best_iou, best_idx = 0.0, -1
            for i in range(N):
                bb = kpts_to_bbox_xyxy(kxy_all[i])
                if bb is None: continue
                giou = iou_xyxy((bb[0]+ex1, bb[1]+ey1, bb[2]+ex1, bb[3]+ey1), (x1, y1, x2, y2))
                if giou > best_iou:
                    best_iou, best_idx = giou, i
            if best_idx < 0: continue

            kxy = kxy_all[best_idx].astype(np.float32)
            kxy[:, 0] = np.clip(kxy[:, 0] + ex1, 0, W - 1)
            kxy[:, 1] = np.clip(kxy[:, 1] + ey1, 0, H - 1)
            draw_skeleton(frame, kxy)

        frame_count += 1
        elapsed = max(1e-6, time.perf_counter() - t0)
        fps = frame_count / elapsed
        
        cv2.putText(frame, f"FPS: {fps:.1f} | HP: {len(high_priority_peds)}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 230, 40), 2, cv2.LINE_AA)

        prev_detections = current_bboxes

        current_track_ids = set(track_id_map.values())
        expired_ids = [tid for tid in list(kalman_filters.keys()) if tid not in current_track_ids]
        for tid in expired_ids:
            del kalman_filters[tid]
            del distance_kf_filters[tid]
            del track_histories[tid]
            del track_states[tid]

        view = cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE)
        cv2.imshow(WINDOW_NAME, view)
        
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if ENABLE_DEBUG_LOGGING:
        csv_file.close()
        
        print(f"\n[INFO] Debug log saved to {csv_filename}")
    
    print(f"\n[INFO] 완료. 총 {frame_count} 프레임")

if __name__ == "__main__":
    main()
