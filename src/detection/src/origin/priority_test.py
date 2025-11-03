import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/home/jaemin/yolopose/model/yolo11s-pose.pt")
video_path = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4" # 6,9
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit("error: cannot open video")

# YOLO
IMGSZ = 960
CONF_THR = 0.2
IOU_THR  = 0.5
PERSON_CLASS = 0 

# grid
GRID_ROWS, GRID_COLS = 64, 64
SELECT_R1, SELECT_R2 = 38, 54
SELECT_C1, SELECT_C2 = 16, 54

# vizualization
DRAW_GRID = True
GRID_COLOR = (255, 255, 255)
GRID_THICK = 1

# priority threshold
TH_NORMAL = 0.25
TH_DANGER = 0.35

# BGR
CLR_SAFE   = (0, 200, 0)   
CLR_NORMAL = (0, 165, 255) 
CLR_DANGER = (0, 0, 255)   

def grid_to_pixel_roi(h, w, rows, cols, r1, r2, c1, c2):
    cell_h = h / rows
    cell_w = w / cols
    y1 = int((r1 - 1) * cell_h); y2 = int(r2 * cell_h)
    x1 = int((c1 - 1) * cell_w); x2 = int(c2 * cell_w)
    y1 = max(0, min(h-1, y1)); y2 = max(1, min(h, y2))
    x1 = max(0, min(w-1, x1)); x2 = max(1, min(w, x2))
    return x1, y1, x2, y2

def inter_box(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
    ix1 = max(x1, rx1); iy1 = max(y1, ry1)
    ix2 = min(x2, rx2); iy2 = min(y2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2

def box_bottom_center_xyxy(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int(y2)

def draw_grid_on_crop(crop_img, rx1, ry1, full_h, full_w,
                      rows, cols, r1, r2, c1, c2,
                      color=(255,255,255), thick=1):
    ch, cw = crop_img.shape[:2]
    cell_h = full_h / rows
    cell_w = full_w / cols

    for r in range(r1, r2):
        y_full = int(r * cell_h)
        y = y_full - ry1
        if 0 <= y < ch:
            cv2.line(crop_img, (0, y), (cw-1, y), color, thick)
    for c in range(c1, c2):
        x_full = int(c * cell_w)
        x = x_full - rx1
        if 0 <= x < cw:
            cv2.line(crop_img, (x, 0), (x, ch-1), color, thick)

def classify_risk(by, y_car, y_max):
    denom = max(1, (y_max - y_car))
    d = (by - y_car) / denom
    d = max(0.0, min(1.0, d))
    if d >= TH_DANGER:
        return "danger", CLR_DANGER
    elif d >= TH_NORMAL:
        return "normal", CLR_NORMAL
    else:
        return "safe", CLR_SAFE

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    rx1, ry1, rx2, ry2 = grid_to_pixel_roi(
        H, W, GRID_ROWS, GRID_COLS, SELECT_R1, SELECT_R2, SELECT_C1, SELECT_C2
    )

    # 차량 기준선 y_car: 선택영역의 상단 y(=R38 상단 경계)
    y_car = ry1
    y_max = H - 1  # 정규화 상한(프레임 하단)

    # 전체 프레임에서 탐지
    results = model.predict(
        frame, imgsz=IMGSZ, conf=CONF_THR, iou=IOU_THR, classes=[PERSON_CLASS], verbose=False
    )
    r = results[0]
    boxes = r.boxes

    # 선택 영역 crop
    crop = frame[ry1:ry2, rx1:rx2].copy()

    # 선택 영역과 겹치는 박스만 표시 + 위험 등급
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy
        confs = boxes.conf
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i].tolist())
            inter = inter_box(x1, y1, x2, y2, rx1, ry1, rx2, ry2)
            if inter is None:
                continue

            # 하단 중앙 기준으로 근접도 평가
            bx, by = box_bottom_center_xyxy(x1, y1, x2, y2)
            if not (rx1 <= bx < rx2 and ry1 <= by < ry2):
                # 하단 중앙이 선택영역 밖이면 제외
                continue

            # 위험 등급 산출
            risk_label, color = classify_risk(by, y_car, y_max)

            # crop 좌표계로 변환하여 그리기(교집합 박스 사용)
            ix1, iy1, ix2, iy2 = inter
            cx1, cy1 = ix1 - rx1, iy1 - ry1
            cx2, cy2 = ix2 - rx1, iy2 - ry1
            conf = float(confs[i].item()) if confs is not None else 0.0

            cv2.rectangle(crop, (cx1, cy1), (cx2, cy2), color, 2)
            label = f"{risk_label} {conf:.2f}"
            cv2.putText(crop, label, (cx1, max(0, cy1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # 하단 중앙 점 찍기
            cbx, cby = bx - rx1, by - ry1
            cv2.circle(crop, (cbx, cby), 3, color, -1)

    # 선택영역 그리드 시각화
    if DRAW_GRID:
        draw_grid_on_crop(crop, rx1, ry1, H, W,
                          GRID_ROWS, GRID_COLS,
                          SELECT_R1, SELECT_R2, SELECT_C1, SELECT_C2,
                          color=GRID_COLOR, thick=GRID_THICK)

    cv2.imshow(f"Crop R{SELECT_R1}-{SELECT_R2}, C{SELECT_C1}-{SELECT_C2}", crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
