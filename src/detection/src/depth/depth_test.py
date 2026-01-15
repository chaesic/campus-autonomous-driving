import cv2
import torch
import numpy as np
import os

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ENCODER = 'vits' 
VIDEO_PATH = '/home/jaemin/JAAD/JAAD_clips/video_0006.mp4'
SCALE_PERCENT = 50 

# 본넷 제거를 위한 하단 크롭 비율
CROP_BOTTOM_PERCENT = 20

# 너무 가까운 지면 등을 무시 (값을 올릴수록 더 먼 곳부터 표현)
CLIP_BOTTOM_PERCENTILE = 5

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

print(f"Using device: {DEVICE}")
model = DepthAnythingV2(**model_configs[ENCODER])

model_path = f'~/yolopose/model/depth_anything_v2_{ENCODER}.pth'
expanded_path = os.path.expanduser(model_path)
print(f"Loading model from: {expanded_path}")

model.load_state_dict(torch.load(expanded_path, map_location='cpu'))
model = model.to(DEVICE).eval()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
crop_pixels = int(orig_height * CROP_BOTTOM_PERCENT / 100)
display_dim = (int(orig_width * SCALE_PERCENT / 100), int(orig_height * SCALE_PERCENT / 100))

inference_width = int(orig_width / 2)
inference_height = int((orig_height - crop_pixels) / 2)
inference_dim = (inference_width, inference_height)
print(f"Inference resolution: {inference_width}x{inference_height}")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Finished processing video or failed to grab frame.")
        break
    
    frame_cropped = frame[:orig_height - crop_pixels, :]
    
    frame_for_inference = cv2.resize(frame_cropped, inference_dim, interpolation=cv2.INTER_AREA)
    
    depth = model.infer_image(frame_for_inference)
    
    # 최소 깊이 값만 설정하여 가까운 지면 무시
    vmin = np.percentile(depth, CLIP_BOTTOM_PERCENTILE)
    # 최대 깊이는 그대로 사용하므로 vmax는 np.max(depth)가 됨
    depth_clipped = np.clip(depth, vmin, np.max(depth))
    
    depth_normalized = cv2.normalize(depth_clipped, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    
    frame_resized = cv2.resize(frame, display_dim, interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(depth_colored, (display_dim[0], int(display_dim[1] * (orig_height - crop_pixels)/orig_height)), interpolation=cv2.INTER_AREA)
    
    cv2.imshow('Original Video', frame_resized)
    cv2.imshow('Depth Video', depth_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Exiting program.")
cap.release()
cv2.destroyAllWindows()