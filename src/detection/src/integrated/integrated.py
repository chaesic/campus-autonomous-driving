import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PedestrianPerceptionModule:
    """
    매 프레임마다:
    1. 보행자 검출 (YOLO)
    2. 깊이 맵 생성
    3. 거리 측정
    4. 우선순위 필터링
    5. Pose 추출
    6. Annotation 수집
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.det_model = YOLO(config['det_model_path'])
        self.pose_model = YOLO(config['pose_model_path'])

        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384]
            }
        }
        self.depth_model = DepthAnythingV2(**model_configs[config['depth_encoder']])
        depth_path = os.path.expanduser(
            f"{config['depth_model_dir']}/depth_anything_v2_{config['depth_encoder']}.pth"
        )
        self.depth_model.load_state_dict(torch.load(depth_path, map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()
        
        self.jaad = self._load_jaad_annotations(config['jaad_dir'])

        self.cached_depth = None
        self.frame_count = 0
        
    def _load_jaad_annotations(self, jaad_dir):
        jaad_data = {}
        return jaad_data
    
    def get_bbox_distance(self, bbox, depth_map, orig_height):
        # BBox 영역에서 거리 추출 
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Depth map 크기에 맞게 좌표 변환
        scale_y = depth_map.shape[0] / orig_height
        
        y1_d = max(0, int(y1 * scale_y))
        y2_d = min(depth_map.shape[0], int(y2 * scale_y))
        x1_d = max(0, int(x1 * scale_y))
        x2_d = min(depth_map.shape[1], int(x2 * scale_y))
        
        roi_depth = depth_map[y1_d:y2_d, x1_d:x2_d]
        
        if roi_depth.size == 0:
            return np.inf
        
        # 노이즈 제거
        lower = np.percentile(roi_depth, 5)
        upper = np.percentile(roi_depth, 95)
        valid_depths = roi_depth[(roi_depth >= lower) & (roi_depth <= upper)]
        
        if len(valid_depths) == 0:
            return np.median(roi_depth)
        
        return np.median(valid_depths)
    
    def process_frame(self, frame, video_id=None, frame_id=None):
        self.frame_count += 1
        orig_height, orig_width = frame.shape[:2]

        det_results = self.det_model.predict(
            frame, imgsz=self.config['img_size_det'],
            conf=self.config['conf_det'], iou=self.config['iou_det'],
            classes=[0], verbose=False, device=self.device
        )
        bboxes = det_results[0].boxes.xyxy.cpu().numpy()
        
        if self.frame_count % (self.config['depth_skip_frames'] + 1) == 1:
            crop_pixels = int(orig_height * self.config['crop_bottom_percent'] / 100)
            frame_cropped = frame[:orig_height - crop_pixels, :]
            
            inference_dim = (
                int(orig_width * self.config['depth_scale']),
                int((orig_height - crop_pixels) * self.config['depth_scale'])
            )
            frame_for_inference = cv2.resize(
                frame_cropped, inference_dim, interpolation=cv2.INTER_AREA
            )
            self.cached_depth = self.depth_model.infer_image(frame_for_inference)
        
        distances = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            dist = self.get_bbox_distance(bbox, self.cached_depth, orig_height)
            distances.append(dist)
            
            if self.config['min_distance'] < dist < self.config['max_distance']:
                valid_indices.append(i)
        
        valid_bboxes = bboxes[valid_indices]
        valid_distances = [distances[i] for i in valid_indices]
        
        pedestrians = []
        
        if len(valid_bboxes) > 0:
            crops, expanded_coords = [], []
            for bbox in valid_bboxes:
                bbox = bbox.astype(int)
                
                box_w = bbox[2] - bbox[0]
                box_h = bbox[3] - bbox[1]
                if box_w * box_h < self.config['min_detection_area']:
                    continue
                
                ex1 = max(0, int(bbox[0] - box_w * 0.3))
                ex2 = min(orig_width, int(bbox[2] + box_w * 0.3))
                ey1 = max(0, int(bbox[1] - box_h * 0.3))
                ey2 = min(orig_height, int(bbox[3] + box_h * 0.3))
                
                crop = frame[ey1:ey2, ex1:ex2]
                if crop.size > 0:
                    crops.append(crop)
                    expanded_coords.append((ex1, ey1))
            
            if crops:
                pose_results = self.pose_model.predict(
                    crops, imgsz=self.config['img_size_pose'],
                    conf=self.config['pose_conf_thr'],
                    verbose=False, device=self.device
                )
                
                for i, res in enumerate(pose_results):
                    if res.keypoints is None:
                        continue
                    
                    kxy_all = res.keypoints.xy.cpu().numpy()
                    if kxy_all.ndim != 3 or kxy_all.shape[0] == 0:
                        continue
                    
                    kxy = kxy_all[0].astype(np.float32)  # (18, 2)
                    kxy[:, 0] += expanded_coords[i][0]
                    kxy[:, 1] += expanded_coords[i][1]
                    
                    annotation = self._get_annotation(video_id, frame_id)
                    
                    pedestrians.append({
                        'ped_id': f"P_{len(pedestrians)}",
                        'bbox': valid_bboxes[i],
                        'pose': kxy,  # (18, 2)
                        'distance': valid_distances[i],
                        'annotation': annotation
                    })
        
        return {
            'frame_id': frame_id or self.frame_count,
            'pedestrians': pedestrians
        }
    
    def _get_annotation(self, video_id, frame_id):
        """JAAD에서 Annotation 추출"""
        return {
            'crossing': False, 
            'looking': False,
            'walking': True
        }

class TrajectoryDataPreparation:
    """
    과거 프레임들의 Pose + Distance + Annotation으로부터
    학습 데이터셋 생성
    """
    
    def __init__(self, config):
        self.config = config
        self.sequences = defaultdict(list)  # {ped_id: [frame_data, ...]}
        
    def add_frame_data(self, frame_data):
        """프레임 데이터 추가"""
        for ped in frame_data['pedestrians']:
            ped_id = ped['ped_id']
            self.sequences[ped_id].append({
                'pose': ped['pose'],  # (18, 2)
                'distance': ped['distance'],
                'annotation': ped['annotation'],
                'frame_id': frame_data['frame_id']
            })
    
    def create_training_samples(self):
        """학습 샘플 생성"""
        history_len = self.config['history_frames']  # 15
        future_len = self.config['future_frames']    # 12
        
        training_samples = []
        
        for ped_id, sequence in self.sequences.items():
            for t in range(len(sequence) - history_len - future_len):
                history = sequence[t:t + history_len]
                future = sequence[t + history_len:t + history_len + future_len]
                
                if len(history) < history_len or len(future) < future_len:
                    continue
                
                input_sequence = self._prepare_input(history)  # (15, 38)
                future_poses = np.array([f['pose'] for f in future])  # (12, 18, 2)
                future_poses = future_poses.reshape(future_len, -1)   # (12, 36)

                future_annotations = np.array([
                    self._annotation_to_vector(f['annotation'])
                    for f in future
                ])  # (12, num_classes)
                
                training_samples.append({
                    'input': input_sequence,        # (15, 38)
                    'future_pose': future_poses,    # (12, 36)
                    'future_annotation': future_annotations,  # (12, num_classes)
                    'ped_id': ped_id
                })
        
        return training_samples
    
    def _prepare_input(self, history):
        history_len = len(history)
        input_seq = np.zeros((history_len, 38))
        
        for t, frame_data in enumerate(history):
            pose = frame_data['pose']  # (18, 2)
            pose_flat = pose.reshape(-1)  # (36,)
            
            distance = frame_data['distance']  # scalar
            
            annotation_vec = self._annotation_to_vector(
                frame_data['annotation']
            )  # (1,) or (num_classes,)

            input_seq[t] = np.concatenate([
                pose_flat,
                [distance],
                annotation_vec[:1] 
            ])
        
        return input_seq
    
    def _annotation_to_vector(self, annotation):
        crossing = 1.0 if annotation.get('crossing', False) else 0.0
        return np.array([crossing])
    
    def save_sequences(self, save_path):
        samples = self.create_training_samples()
        split_idx = int(len(samples) * 0.8)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        np.savez(
            os.path.join(save_path, 'train_sequences.npz'),
            inputs=np.array([s['input'] for s in train_samples]),
            future_poses=np.array([s['future_pose'] for s in train_samples]),
            future_annotations=np.array([s['future_annotation'] for s in train_samples])
        )
        
        np.savez(
            os.path.join(save_path, 'val_sequences.npz'),
            inputs=np.array([s['input'] for s in val_samples]),
            future_poses=np.array([s['future_pose'] for s in val_samples]),
            future_annotations=np.array([s['future_annotation'] for s in val_samples])
        )
        
        print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

class MultiTaskTrajectoryPredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Linear(38, 64)
        self.lstm_encoder = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.pose_decoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.pose_output = nn.Linear(128, 36)

        self.annotation_decoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )
        self.annotation_output = nn.Linear(64, 2)  # binary: crossing or not
        
    def forward(self, x):

        batch_size = x.shape[0]

        embedded = self.embedding(x)  # (batch, 15, 64)

        encoder_out, (h_n, c_n) = self.lstm_encoder(embedded)  # (batch, 15, 128)
        pose_out, _ = self.pose_decoder_lstm(
            encoder_out.unsqueeze(1).expand(-1, 12, -1)  # (batch, 12, 128)
        )
        pose_pred = self.pose_output(pose_out)  # (batch, 12, 36)
        
        annotation_out, _ = self.annotation_decoder_lstm(
            encoder_out.unsqueeze(1).expand(-1, 12, -1)  # (batch, 12, 128)
        )
        annotation_pred = self.annotation_output(annotation_out)  # (batch, 12, 2)
        
        return {
            'pose': pose_pred,
            'annotation': annotation_pred
        }

class TrajectoryDataset(Dataset):
    
    def __init__(self, data_path):
        data = np.load(data_path)
        self.inputs = torch.FloatTensor(data['inputs'])
        self.future_poses = torch.FloatTensor(data['future_poses'])
        self.future_annotations = torch.FloatTensor(data['future_annotations'])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'future_pose': self.future_poses[idx],
            'future_annotation': self.future_annotations[idx]
        }


class TrajectoryTrainer:
 
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MultiTaskTrajectoryPredictor(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.pose_loss_fn = nn.MSELoss()
        self.annotation_loss_fn = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs = batch['input'].to(self.device)  # (B, 15, 38)
            future_poses = batch['future_pose'].to(self.device)  # (B, 12, 36)
            future_annotations = batch['future_annotation'].to(self.device)  # (B, 12, 1)
            
            outputs = self.model(inputs)
            pose_pred = outputs['pose']  # (B, 12, 36)
            annotation_pred = outputs['annotation']  # (B, 12, 2)
            
            pose_loss = self.pose_loss_fn(pose_pred, future_poses)

            annotation_loss = 0
            for t in range(future_annotations.shape[1]):
                target = future_annotations[:, t, :].long().squeeze(1)
                pred = annotation_pred[:, t, :]
                annotation_loss += self.annotation_loss_fn(pred, target)
            annotation_loss /= future_annotations.shape[1]

            alpha = self.config.get('alpha', 0.7) 
            beta = self.config.get('beta', 0.3)  
            total = alpha * pose_loss + beta * annotation_loss

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            
            total_loss += total.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                future_poses = batch['future_pose'].to(self.device)
                future_annotations = batch['future_annotation'].to(self.device)
                
                outputs = self.model(inputs)
                pose_pred = outputs['pose']
                annotation_pred = outputs['annotation']
                
                pose_loss = self.pose_loss_fn(pose_pred, future_poses)
                
                annotation_loss = 0
                for t in range(future_annotations.shape[1]):
                    target = future_annotations[:, t, :].long().squeeze(1)
                    pred = annotation_pred[:, t, :]
                    annotation_loss += self.annotation_loss_fn(pred, target)
                annotation_loss /= future_annotations.shape[1]
                
                alpha = self.config.get('alpha', 0.7)
                beta = self.config.get('beta', 0.3)
                total = alpha * pose_loss + beta * annotation_loss
                
                total_loss += total.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=50):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("Training complete")
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # 설정
    config = {

        'det_model_path': "/home/jaemin/yolopose/model/yolo11s.pt",
        'pose_model_path': "/home/jaemin/yolopose/model/yolo11s-pose.pt",
        'depth_model_dir': "~/yolopose/model/",
        'depth_encoder': 'vits',
        'jaad_dir': "/home/jaemin/JAAD/",
        'video_path': "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4",
        
        'img_size_det': 1280,
        'img_size_pose': 384,
        'conf_det': 0.4,
        'iou_det': 0.5,
        'pose_conf_thr': 0.01,
        'min_detection_area': 3000,
        'crop_bottom_percent': 20,
        'depth_scale': 0.5,
        'depth_skip_frames': 2,
        'min_distance': 1.0,
        'max_distance': 30.0,

        'history_frames': 15,
        'future_frames': 12,

        'batch_size': 32,
        'epochs': 50,
        'alpha': 0.7, 
        'beta': 0.3,  
    }

    print("=" * 80)
    print("Perception Module")
    print("=" * 80)
    
    perception = PedestrianPerceptionModule(config)
    data_prep = TrajectoryDataPreparation(config)
    
    cap = cv2.VideoCapture(config['video_path'])
    frame_count = 0
    
    while True and frame_count < 300:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = perception.process_frame(frame)
        data_prep.add_frame_data(frame_data)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, "
                  f"Pedestrians detected: {len(frame_data['pedestrians'])}")
    
    cap.release()

    print("\n" + "=" * 80)
    print("Data Preparation")
    print("=" * 80)
    
    os.makedirs("./data", exist_ok=True)
    data_prep.save_sequences("./data")

    print("\n" + "=" * 80)
    print("Multi-Task Learning")
    print("=" * 80)
    
    train_dataset = TrajectoryDataset("./data/train_sequences.npz")
    val_dataset = TrajectoryDataset("./data/val_sequences.npz")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    trainer = TrajectoryTrainer(config)
    trainer.train(train_loader, val_loader, epochs=config['epochs'])
    trainer.save_model("./model/trajectory_predictor.pth")
    
    print("\n" + "=" * 80)
    print("Complete")
    print("=" * 80)
