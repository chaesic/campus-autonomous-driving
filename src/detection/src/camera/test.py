import cv2
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 모델 경로
DET_MODEL = "/home/jaemin/yolopose/model/yolo11s.pt"
POSE_MODEL = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"

# 🔹 임계값 (민감도 높임)
TH_NORMAL = 0.15
TH_DANGER = 0.22

from ultralytics import YOLO

print(f"🔹 Using device: {DEVICE}")
print(f"🔹 Loading detection model from {DET_MODEL}")
yolo_model = YOLO(DET_MODEL).to(DEVICE)

print(f"🔹 Loading pose model from {POSE_MODEL}")
pose_model = YOLO(POSE_MODEL).to(DEVICE)

# 🔹 거리 추정 함수 (상대적 위험도 기반)
def estimate_relative_distance(bbox, frame_height):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    cy = y2 / frame_height  # 하단부 위치 비율 (0~1)
    size_factor = h / frame_height  # bbox 높이 비율 (0~1)
    # 화면 하단에 있고, bbox가 크면 더 가까움 (값 작을수록 위험)
    distance_score = 1.0 - (0.6 * cy + 0.4 * size_factor)
    return max(0.0, min(1.0, distance_score))  # 0~1 사이 정규화

# 🔹 위험도 판정
def get_pedestrian_state(distance_score):
    if distance_score < TH_NORMAL:
        return "danger"
    elif distance_score < TH_DANGER:
        return "normal"
    else:
        return "safe"

# 🔹 색상 정의
COLORS = {"safe": (0, 255, 0), "normal": (0, 165, 255), "danger": (0, 0, 255)}

# 🔹 Pose skeleton 연결 정의 (COCO keypoints 기반 예시)
POSE_CONNECTIONS = [
    (5, 7), (7, 9),   # 왼팔
    (6, 8), (8, 10),  # 오른팔
    (11, 13), (13, 15),  # 왼다리
    (12, 14), (14, 16),  # 오른다리
    (5, 6), (11, 12),  # 어깨, 엉덩이
    (5, 11), (6, 12)   # 몸통 연결
]

# 🔹 skeleton 그리기
def draw_skeleton(frame, keypoints, color):
    for (x, y, conf) in keypoints:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    for (i, j) in POSE_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (960, 540))
    results = yolo_model(frame_resized)
    detections = results.xyxy[0].cpu().numpy()

    # ✅ 보행자만 필터링 후 상위 5명만 유지
    pedestrians = [d for d in detections if int(d[5]) == 0][:5]

    for det in pedestrians:
        x1, y1, x2, y2, conf, cls = det
        bbox = [int(x1), int(y1), int(x2), int(y2)]

        distance_score = estimate_relative_distance(bbox, frame_resized.shape[0])
        state = get_pedestrian_state(distance_score)

        color = COLORS[state]
        cv2.rectangle(frame_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"{state.upper()} ({distance_score:.2f})"
        cv2.putText(frame_resized, label, (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ✅ normal/danger 보행자는 pose 추출 및 skeleton 시각화
        if state in ["normal", "danger"]:
            crop = frame_resized[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            pose_result = pose_model(crop)

            if hasattr(pose_result, 'keypoints'):
                keypoints = pose_result.keypoints[0].cpu().numpy()
                # bbox offset 적용
                keypoints[:, 0] += bbox[0]
                keypoints[:, 1] += bbox[1]
                draw_skeleton(frame_resized, keypoints, color)

    cv2.imshow("Pedestrian Risk Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
