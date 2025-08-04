"""
Valorant Video Tracker with Motion Prediction and Teammate-based Inference
Handles occlusion and missed detections through intelligent prediction
"""

import json
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
import math
from ml_models.agent_classifier_cnn import AgentClassifierCNN
from ml_models.orientation_cnn import OrientationCNN
from tracker.motion_predictor import MotionPredictor
from tracker.agent_group_manager import AgentGroupManager

class ValorantVideoTracker:
    """Tracker with motion prediction and teammate-based inference"""
    
    def __init__(self, yolo_model_path, cnn_model_path, orientation_model_path, device='cuda', use_agent_grouping=False):
        """Initialize tracker with models and configuration
        
        Args:
            yolo_model_path: Path to YOLO model for agent detection
            cnn_model_path: Path to CNN model for agent classification  
            orientation_model_path: Path to CNN model for orientation prediction
            device: 'cuda' or 'cpu'
            use_agent_grouping: Whether to use agent group manager for overlapping detections
        """
        from ultralytics import YOLO
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load CNN classification model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(cnn_model_path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.input_size = checkpoint['input_size']
        self.classes = checkpoint['classes']
        
        self.cnn_model = AgentClassifierCNN(
            num_classes=self.num_classes,
            input_size=self.input_size
        )
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # Load orientation model
        orient_checkpoint = torch.load(orientation_model_path, map_location=self.device)
        self.orient_input_size = orient_checkpoint.get('input_size', 64)
        
        self.orientation_model = OrientationCNN(input_size=self.orient_input_size)
        self.orientation_model.load_state_dict(orient_checkpoint['model_state_dict'])
        self.orientation_model.to(self.device)
        self.orientation_model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.orient_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.orient_input_size, self.orient_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # YOLO class names mapping
        self.yolo_classes = {
            0: "team1", 1: "team2", 2: "team1-trip-rect", 3: "team1-trip-circle",
            4: "team1-trip-line", 5: "team2-trip-rect", 6: "team2-trip-circle",
            7: "team2-trip-line", 8: "team1-turret", 9: "team2-turret",
            10: "wall-rect", 11: "wall-circle", 12: "wall-line",
            13: "smoke-line", 14: "smoke-circle"
        }
        
        # Color scheme
        self.colors = {
            # Agents
            "team1": (0, 255, 0),       # Green
            "team2": (0, 0, 255),       # Red

            # Team utilities
            "team1-turret": (200, 255, 200),        # Dark green
            "team2-turret": (200, 200, 255),        # Dark red
            "team1-trip-circle": (100, 255, 100),   # Darkish green
            "team1-trip-rect": (100, 255, 100),
            "team1-trip-line": (100, 255, 100),
            "team2-trip-circle": (100, 100, 255),   # Darkish red
            "team2-trip-rect": (100, 100, 255),
            "team2-trip-line": (100, 100, 255),
            
            # Neutral utilities
            "smoke-circle": (200, 200, 200),    # Gray
            "smoke-line": (200, 200, 200), 
            "wall-rect": (255, 255, 0),         # Yellow
            "wall-circle": (255, 255, 0),
            "wall-line": (255, 255, 0),
        }
        
        # tracking parameters
        self.tracks = {}
        self.next_track_id = 0
        self.max_age = 90  # Increased from  (1.5 seconds at 60fps)
        self.death_threshold = 120  # 2 seconds at 60fps - assume dead after this
        self.min_hits = 3
        self.iou_threshold = 0.3
        self.confidence_threshold = 0.70
        self.prediction_threshold = 0.4  # Lower threshold for predicted positions
        
        # Motion predictor
        self.motion_predictor = MotionPredictor()
        
        # Agent group manager (optional)
        self.use_agent_grouping = use_agent_grouping
        if self.use_agent_grouping: self.agent_group_manager = AgentGroupManager(overlap_threshold=0.5,)
        
        # Track states
        self.TRACK_STATES = {
            'DETECTED': 'detected',      # Currently detected
            'PREDICTED': 'predicted',    # Position predicted
            'DEAD': 'dead'              # Assumed dead
        }

        # Calibration phase parameters
        self.calibration_frames = 300  # Number of frames for calibration (3 seconds at 60 fps)
        self.agent_detection_counts = defaultdict(lambda: defaultdict(int))  # team -> agent -> count
        self.valid_agents = {}  # team -> set of valid agent names
        self.calibration_complete = False
    
    def detect_and_classify_frame(self, frame, conf_threshold=0.5, padding=5):
        """Detect, classify agents and predict orientation in a single frame"""
        # Run YOLO detection
        results = self.yolo_model(frame, conf=conf_threshold, verbose=False)
        
        h, w = frame.shape[:2]
        detections = {'agents': [], 'utilities': []}
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo_classes.get(class_id, f"unknown_{class_id}")
                yolo_conf = float(box.conf[0].cpu().numpy())
                
                if class_id in [0, 1]:  # Agent classes
                    x1_pad = max(0, int(x1 - padding))
                    y1_pad = max(0, int(y1 - padding))
                    x2_pad = min(w, int(x2 + padding))
                    y2_pad = min(h, int(y2 + padding))
                    
                    crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # Classify agent and predict orientation
                    agent_name, cnn_conf = self._classify_agent(crop)
                    orientation, orient_conf = self._predict_orientation(crop)
                    combined_conf = yolo_conf * cnn_conf
                    
                    if combined_conf >= self.confidence_threshold:
                        detections['agents'].append({
                            'position': {'x': int((x1 + x2) / 2), 'y': int((y1 + y2) / 2)},
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'team': class_name,
                            'agent': agent_name,
                            'orientation': orientation,
                            'confidence': {
                                'detection': yolo_conf,
                                'classification': cnn_conf,
                                'orientation': orient_conf,
                                'combined': combined_conf
                            },
                            'frame': frame
                        })
        
        # Apply agent group manager if enabled
        if self.use_agent_grouping and detections['agents']:
            detections['agents'] = self.agent_group_manager.process_overlapping_agents(
                detections['agents']
            )
        
        # Process utility detections
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo_classes.get(class_id, f"unknown_{class_id}")
                yolo_conf = float(box.conf[0].cpu().numpy())
                
                if class_id not in [0, 1]:  # Utility classes
                    detections['utilities'].append({
                        'position': {'x': int((x1 + x2) / 2), 'y': int((y1 + y2) / 2)},
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'type': class_name,
                        'confidence': yolo_conf
                    })
        
        return detections
    
    def _classify_agent(self, crop):
        """Classify agent using CNN"""
        if crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop
            
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.cnn_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        return self.classes[pred.item()], conf.item()
    
    def _predict_orientation(self, crop):
        """Predict agent orientation using orientation CNN"""
        if crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop
            
        img_tensor = self.orient_transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.orientation_model(img_tensor)
            
        sin_theta = output[0, 0].cpu().item()
        cos_theta = output[0, 1].cpu().item()
        
        angle_rad = np.arctan2(sin_theta, cos_theta)
        angle_deg = (angle_rad * 180 / np.pi + 360) % 360
        
        magnitude = np.sqrt(sin_theta**2 + cos_theta**2)
        confidence = magnitude
        
        return angle_deg, confidence
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_to_tracks(self, detections, predicted_positions):
        """matching including predicted positions"""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        active_track_ids = [tid for tid, track in self.tracks.items() 
                           if track['state'] != self.TRACK_STATES['DEAD']]
        
        if not active_track_ids:
            return [], list(range(len(detections))), []
        
        # Build cost matrix
        cost_matrix = np.zeros((len(active_track_ids), len(detections)))
        
        for i, track_id in enumerate(active_track_ids):
            track = self.tracks[track_id]
            predicted_pos = predicted_positions.get(track_id)
            
            for j, det in enumerate(detections):
                # Distance cost (use predicted position if available)
                if predicted_pos:
                    pos_cost = euclidean(
                        [predicted_pos['x'], predicted_pos['y']],
                        [det['position']['x'], det['position']['y']]
                    ) / 100.0  # Normalize
                else:
                    last_pos = track['position_history'][-1]
                    pos_cost = euclidean(
                        [last_pos['x'], last_pos['y']],
                        [det['position']['x'], det['position']['y']]
                    ) / 100.0
                
                # Agent and team consistency costs
                agent_cost = 0 if track['agent'] == det['agent'] else 0.5
                team_cost = 0 if track['team'] == det['team'] else 1.0
                
                # Combined cost
                cost_matrix[i, j] = pos_cost + agent_cost + team_cost
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(active_track_ids)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 2.0:  # Threshold for valid match
                matches.append((active_track_ids[r], c))
                unmatched_detections.remove(c)
                unmatched_tracks.remove(active_track_ids[r])
        
        return matches, unmatched_detections, unmatched_tracks
    
    def predict_missing_positions(self, frame_number):
        """Predict positions for tracks without current detections"""
        predicted_positions = {}
        
        for track_id, track in self.tracks.items():
            # Only predict for valid agents
            if (track['state'] == self.TRACK_STATES['DETECTED'] and 
                track['last_seen'] < frame_number and
                track.get('is_valid', True)):  # Added validity check
                
                # Get teammates for influence
                teammates = [t for tid, t in self.tracks.items() 
                        if (t['team'] == track['team'] and 
                            tid != track_id and 
                            t['state'] != self.TRACK_STATES['DEAD'] and
                            t.get('is_valid', True))]  # Only consider valid teammates
                
                # Predict position
                predicted_pos = self.motion_predictor.predict_position(track, teammates)
                predicted_positions[track_id] = predicted_pos
        
        return predicted_positions
    
    def update_tracks(self, agent_detections, frame_number):
        """track updating with motion prediction"""
        # Update calibration
        self.update_calibration(agent_detections, frame_number)
        
        # Predict positions for missing tracks
        predicted_positions = self.predict_missing_positions(frame_number)
        
        # Match detections to tracks (including predicted)
        matches, unmatched_dets, unmatched_tracks = self.match_detections_to_tracks(
            agent_detections, predicted_positions
        )
        
        # Track agent assignments per team
        assigned_agents = defaultdict(set)
        
        # Update matched tracks
        for track_id, det_idx in matches:
            det = agent_detections[det_idx]
            track = self.tracks[track_id]
            
            # Update validity based on calibration
            track['is_valid'] = self.is_agent_valid(det['team'], det['agent'])
            
            # Update track with detection
            track['position_history'].append(det['position'])
            if len(track['position_history']) > 30:
                track['position_history'].pop(0)
            
            track['orientation_history'].append(det['orientation'])
            if len(track['orientation_history']) > 30:
                track['orientation_history'].pop(0)
            
            track['bbox'] = det['bbox']
            track['last_seen'] = frame_number
            track['hits'] += 1
            track['age'] = 0
            track['state'] = self.TRACK_STATES['DETECTED']
            track['confidence'] = det['confidence']
            track['agent'] = det['agent']
            track['orientation'] = det['orientation']
            
            # Only count valid agents in assignments
            if track['is_valid']:
                assigned_agents[track['team']].add(track['agent'])
        
        # Handle unmatched tracks (predict positions)
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            
            # Only process valid tracks for prediction
            if track_id in predicted_positions and track.get('is_valid', True):
                # Use predicted position
                predicted_pos = predicted_positions[track_id]
                track['position_history'].append(predicted_pos)
                if len(track['position_history']) > 30:
                    track['position_history'].pop(0)
                
                # Keep last known orientation or predict from movement
                if len(track['position_history']) >= 2:
                    # Calculate movement direction as fallback orientation
                    last_pos = track['position_history'][-2]
                    curr_pos = track['position_history'][-1]
                    dx = curr_pos['x'] - last_pos['x']
                    dy = curr_pos['y'] - last_pos['y']
                    if dx != 0 or dy != 0:
                        movement_angle = (math.atan2(dy, dx) * 180 / math.pi + 360) % 360
                        track['orientation'] = movement_angle
                        track['orientation_history'].append(movement_angle)
                        if len(track['orientation_history']) > 30:
                            track['orientation_history'].pop(0)
                
                track['state'] = self.TRACK_STATES['PREDICTED']
                track['hits'] += 0.5  # Partial hit for prediction
                
                # Update bbox based on predicted position (estimate size)
                pred_x, pred_y = predicted_pos['x'], predicted_pos['y']
                bbox_size = 20  # Estimated agent size
                track['bbox'] = [
                    pred_x - bbox_size, pred_y - bbox_size,
                    pred_x + bbox_size, pred_y + bbox_size
                ]
                
                assigned_agents[track['team']].add(track['agent'])
            else:
                # Age the track (both invalid and valid tracks that weren't predicted)
                track['age'] += 1
                
                # Mark as dead if too old
                if track['age'] > self.death_threshold:
                    track['state'] = self.TRACK_STATES['DEAD']
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = agent_detections[det_idx]
            
            # Check validity for new tracks
            is_valid = self.is_agent_valid(det['team'], det['agent'])
            
            # Only create tracks for agents not already assigned (valid agents only)
            if is_valid and det['agent'] not in assigned_agents[det['team']]:
                self.tracks[self.next_track_id] = {
                    'id': self.next_track_id,
                    'agent': det['agent'],
                    'team': det['team'],
                    'bbox': det['bbox'],
                    'orientation': det['orientation'],
                    'position_history': deque([det['position']], maxlen=30),
                    'orientation_history': deque([det['orientation']], maxlen=30),
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'hits': 1,
                    'age': 0,
                    'state': self.TRACK_STATES['DETECTED'],
                    'confidence': det['confidence'],
                    'is_valid': is_valid
                }
                self.next_track_id += 1
                assigned_agents[det['team']].add(det['agent'])
            elif not is_valid:
                # Still create tracks for invalid agents (for visualization)
                # but don't add to assigned_agents
                self.tracks[self.next_track_id] = {
                    'id': self.next_track_id,
                    'agent': det['agent'],
                    'team': det['team'],
                    'bbox': det['bbox'],
                    'orientation': det['orientation'],
                    'position_history': deque([det['position']], maxlen=30),
                    'orientation_history': deque([det['orientation']], maxlen=30),
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'hits': 1,
                    'age': 0,
                    'state': self.TRACK_STATES['DETECTED'],
                    'confidence': det['confidence'],
                    'is_valid': is_valid
                }
                self.next_track_id += 1
        
        # Clean up very old dead tracks
        tracks_to_remove = [
            tid for tid, track in self.tracks.items()
            if (track['state'] == self.TRACK_STATES['DEAD'] and 
                track['age'] > self.max_age * 2)
        ]
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_active_tracks(self, min_hits=None, include_invalid=False):
        """Get currently active tracks (detected or predicted)
        
        Args:
            min_hits: Minimum hits required
            include_invalid: Whether to include invalid agents (for visualization)
        """
        if min_hits is None:
            min_hits = self.min_hits
        
        active_tracks = []
        for track_id, track in self.tracks.items():
            if (track['hits'] >= min_hits and 
                track['state'] != self.TRACK_STATES['DEAD']):
                
                # Skip invalid agents unless specifically requested
                if not include_invalid and not track.get('is_valid', True):
                    continue
                
                active_tracks.append({
                    'track_id': track['id'],
                    'agent': track['agent'],
                    'team': track['team'],
                    'position': track['position_history'][-1],
                    'orientation': track['orientation'],
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'track_length': len(track['position_history']),
                    'state': track['state'],  # Include state for visualization
                    'is_valid': track.get('is_valid', True)  # Include validity
                })
        
        return active_tracks
    
    def visualize_frame_with_tracks(self, frame, tracks, utilities):
        """visualization showing track states and validity"""
        vis_frame = frame.copy()
        
        # Draw utilities first
        for util in utilities:
            x1, y1, x2, y2 = util['bbox']
            util_type = util['type']
            color = self.colors.get(util_type, (255, 255, 255))
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label = util_type
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw agent tracks with state and validity indication
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            team = track['team']
            agent = track['agent']
            track_id = track['track_id']
            orientation = track['orientation']
            state = track.get('state', 'detected')
            is_valid = track.get('is_valid', True)
            
            # Get base color
            base_color = self.colors.get(team, (255, 255, 255))
            
            # Modify color for invalid agents
            if not is_valid:
                # Darken the color and add red tint for invalid agents
                color = (base_color[0] // 3, base_color[1] // 3, min(255, base_color[2] // 3 + 100))
            else:
                color = base_color
            
            # Modify visualization based on state
            if state == self.TRACK_STATES['PREDICTED']:
                # Dashed line for predicted tracks
                self._draw_dashed_rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            else:
                # Solid line for detected tracks
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw X for invalid agents
            if not is_valid:
                # Draw red X across the bounding box
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.line(vis_frame, (x2, y1), (x1, y2), (0, 0, 255), 3)
            
            # Draw orientation arrow
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            arrow_length = min(x2 - x1, y2 - y1) // 2
            angle_rad = orientation * np.pi / 180
            end_x = int(cx + arrow_length * np.cos(angle_rad))
            end_y = int(cy + arrow_length * np.sin(angle_rad))
            
            cv2.arrowedLine(vis_frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
            cv2.circle(vis_frame, (cx, cy), 3, color, -1)
            
            # label with state and validity
            state_symbol = "D" if state == 'detected' else "P"  # D=Detected, P=Predicted
            validity_symbol = "" if is_valid else " [X]"  # X=Invalid/filtered
            label = f"{state_symbol} ID:{track_id} {agent} {orientation:.0f}°{validity_symbol}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            
            # Background color based on state and validity
            if not is_valid:
                bg_color = (0, 0, 100)  # Dark red for invalid
            else:
                bg_color = color if state == 'detected' else (color[0]//2, color[1]//2, color[2]//2)
            
            cv2.rectangle(vis_frame, (x1, label_y - label_size[1] - 4),
                         (x1 + label_size[0], label_y), bg_color, -1)
            cv2.putText(vis_frame, label, (x1, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw a dashed rectangle for predicted tracks"""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Top edge
        for x in range(x1, x2, 10):
            cv2.line(img, (x, y1), (min(x + 5, x2), y1), color, thickness)
        
        # Bottom edge
        for x in range(x1, x2, 10):
            cv2.line(img, (x, y2), (min(x + 5, x2), y2), color, thickness)
        
        # Left edge
        for y in range(y1, y2, 10):
            cv2.line(img, (x1, y), (x1, min(y + 5, y2)), color, thickness)
        
        # Right edge
        for y in range(y1, y2, 10):
            cv2.line(img, (x2, y), (x2, min(y + 5, y2)), color, thickness)
    
    def process_video(self, video_path, output_path=None, output_json=None,
                     conf_threshold=0.5, show_preview=False, save_frequency=30, fps_override=None):
        """
        Process entire video with tracking

        Args:
            fps_override: Override the detected FPS (useful if video FPS is incorrect)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        detected_fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps_override if fps_override is not None else detected_fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Detected FPS: {detected_fps}, Using FPS: {fps}")

        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset tracking and prepare the final results structure
        self.tracks = {}
        self.next_track_id = 0
        self.calibration_complete = False
        self.agent_detection_counts = defaultdict(lambda: defaultdict(int))
        self.valid_agents = {}
        
        # This new 'results' dictionary will be saved to JSON
        results = {
            "video_dimensions": f"{width}x{height}",
            "calibration_frames": self.calibration_frames,
            "valid_agents": {},  # Will be filled after calibration
            "frames": {}
        }

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Agent calibration enabled for first {self.calibration_frames} frames")
        print(f"Death threshold: {self.death_threshold} frames ({self.death_threshold/fps:.1f}s)")

        frame_number = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and classify
            detections = self.detect_and_classify_frame(frame, conf_threshold)

            # tracking update (includes calibration)
            self.update_tracks(detections['agents'], frame_number)

            # Get active tracks - include invalid for visualization, exclude for JSON
            active_tracks_vis = self.get_active_tracks(include_invalid=True)
            active_tracks_json = self.get_active_tracks(include_invalid=False)

            # Store tracking data within the 'frames' key of the results dictionary
            results['frames'][frame_number] = {
                'timestamp': frame_number / fps,
                'agents': active_tracks_json,  # Only valid agents in JSON
                'utilities': detections['utilities'],
                'tracking_stats': {
                    'total_tracks': len(self.tracks),
                    'detected_tracks': len([t for t in active_tracks_vis if t['state'] == 'detected']),
                    'predicted_tracks': len([t for t in active_tracks_vis if t['state'] == 'predicted']),
                    'dead_tracks': len([t for t in self.tracks.values() if t['state'] == 'dead']),
                    'invalid_tracks': len([t for t in active_tracks_vis if not t.get('is_valid', True)])
                }
            }
            
            # Update valid agents in results after calibration
            if frame_number == self.calibration_frames:
                results['valid_agents'] = {team: list(agents) for team, agents in self.valid_agents.items()}

            # Visualize
            if output_path or show_preview:
                vis_frame = self.visualize_frame_with_tracks(frame, active_tracks_vis, detections['utilities'])

                # Add tracking statistics overlay
                self._draw_tracking_stats(vis_frame, results['frames'][frame_number]['tracking_stats'], frame_number)

                if output_path:
                    out.write(vis_frame)

                if show_preview:
                    cv2.imshow('Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_number += 1
            pbar.update(1)

        pbar.close()

        print("Finished processing!!!")
        # Release resources
        cap.release()
        if output_path:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Save final tracking data
        if output_json:
            with open(output_json, 'w') as f:
                # Dump the entire 'results' dictionary
                json.dump(results, f, indent=2)

        # Print final statistics
        self._print_final_stats()

        # Return the structured results
        return results
    
    def update_calibration(self, agent_detections, frame_number):
        """Update agent detection counts during calibration phase"""
        if frame_number < self.calibration_frames:
            for det in agent_detections:
                team = det['team']
                agent = det['agent']
                self.agent_detection_counts[team][agent] += 1
        
        elif frame_number == self.calibration_frames and not self.calibration_complete:
            # Finalize calibration
            self.valid_agents = {}
            
            for team, agent_counts in self.agent_detection_counts.items():
                # Sort agents by detection count
                sorted_agents = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Take top 5 agents or all if less than 5
                if len(sorted_agents) <= 5:
                    self.valid_agents[team] = set(agent for agent, _ in sorted_agents)
                else:
                    self.valid_agents[team] = set(agent for agent, _ in sorted_agents[:5])
                
                print(f"\nCalibration complete for {team}:")
                print(f"  Valid agents: {self.valid_agents[team]}")
                print(f"  Detection counts: {dict(sorted_agents[:5])}")
                if len(sorted_agents) > 5:
                    print(f"  Filtered out: {dict(sorted_agents[5:])}")
            
            self.calibration_complete = True

    def is_agent_valid(self, team, agent):
        """Check if an agent is in the valid list after calibration"""
        if not self.calibration_complete:
            return True  # During calibration, all agents are valid
        
        # If we have valid agents for this team, check if agent is in the list
        if team in self.valid_agents:
            return agent in self.valid_agents[team]
        
        # If no valid agents for this team (shouldn't happen), accept all
        return True
    
    def _draw_tracking_stats(self, frame, stats, frame_number):
        """Draw tracking statistics on frame"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        stats_text = [
            f"Frame: {frame_number}",
            f"Total Tracks: {stats['total_tracks']}",
            f"Detected: {stats['detected_tracks']}",
            f"Predicted: {stats['predicted_tracks']}",
            f"Dead: {stats['dead_tracks']}",
            f"Invalid (filtered): {stats['invalid_tracks']}"
        ]
        
        # Add calibration status
        if not self.calibration_complete:
            stats_text.insert(1, f"CALIBRATING... ({frame_number}/{self.calibration_frames})")
        else:
            stats_text.insert(1, "Calibration Complete")
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            y_offset += 20
    
    def _print_final_stats(self):
        """Print final tracking statistics"""
        total_tracks = len(self.tracks)
        detected_tracks = len([t for t in self.tracks.values() if t['state'] == 'detected'])
        predicted_tracks = len([t for t in self.tracks.values() if t['state'] == 'predicted'])
        dead_tracks = len([t for t in self.tracks.values() if t['state'] == 'dead'])
        invalid_tracks = len([t for t in self.tracks.values() if not t.get('is_valid', True)])

        
        print(f"\nFinal Tracking Statistics:")
        print(f"Total unique tracks created: {total_tracks}")
        print(f"Currently detected: {detected_tracks}")
        print(f"Currently predicted: {predicted_tracks}")
        print(f"Marked as dead: {dead_tracks}")
        print(f"Invalid/filtered: {invalid_tracks}")

        # Team breakdown
        team_stats = defaultdict(lambda: {'detected': 0, 'predicted': 0, 'dead': 0, 'invalid': 0})
        for track in self.tracks.values():
            team_stats[track['team']][track['state']] += 1
            if not track.get('is_valid', True):
                team_stats[track['team']]['invalid'] += 1
        
        print(f"\nPer-team breakdown:")
        for team, stats in team_stats.items():
            print(f"  {team}: Detected={stats['detected']}, Predicted={stats['predicted']}, Dead={stats['dead']}, Invalid={stats['invalid']}")
        
        if self.valid_agents:
            print(f"\nValid agents per team (from calibration):")
            for team, agents in self.valid_agents.items():
                print(f"  {team}: {', '.join(sorted(agents))}")
    
    def analyze_tracking_data(self, tracking_data):
        """analysis including prediction statistics"""
        stats = {
            'total_agent_tracks': self.next_track_id,
            'agents_per_team': defaultdict(set),
            'agent_appearances': defaultdict(int),
            'utility_counts': defaultdict(int),
            'average_track_length': 0,
            'orientation_stats': defaultdict(list),
            'prediction_stats': {
                'frames_with_predictions': 0,
                'total_predictions': 0,
                'prediction_accuracy': 0
            },
            'death_events': [],
            'calibration_info': {
                'calibration_frames': tracking_data.get('calibration_frames', 0),
                'valid_agents': tracking_data.get('valid_agents', {}),
                'filtered_detections': 0
            }
        }
        
        track_lengths = []
        prediction_frames = 0
        total_predictions = 0
        
        # Correct the loop to iterate over tracking_data['frames']
        for frame_num, frame_data in tracking_data['frames'].items():
            frame_predictions = 0
            
            # Analyze agents
            for track in frame_data.get('agents', []):
                stats['agents_per_team'][track['team']].add(track['agent'])
                stats['agent_appearances'][track['agent']] += 1
                stats['orientation_stats'][track['agent']].append(track['orientation'])
                
                if track['track_length'] not in track_lengths:
                    track_lengths.append(track['track_length'])
                
                # Count predictions
                if track.get('state') == 'predicted':
                    frame_predictions += 1
                    total_predictions += 1
            
            if frame_predictions > 0:
                prediction_frames += 1
            
            # Analyze utilities
            for util in frame_data.get('utilities', []):
                stats['utility_counts'][util['type']] += 1
    
        # Convert sets to lists
        stats['agents_per_team'] = {k: list(v) for k, v in stats['agents_per_team'].items()}
        
        # Calculate averages
        if track_lengths:
            stats['average_track_length'] = sum(track_lengths) / len(track_lengths)
        
        # Prediction statistics
        stats['prediction_stats']['frames_with_predictions'] = prediction_frames
        stats['prediction_stats']['total_predictions'] = total_predictions
        if tracking_data['frames']:
            stats['prediction_stats']['prediction_rate'] = prediction_frames / len(tracking_data['frames'])
        else:
            stats['prediction_stats']['prediction_rate'] = 0
        
        # Orientation statistics
        orientation_summary = {}
        for agent, orientations in stats['orientation_stats'].items():
            if orientations:
                orientation_summary[agent] = {
                    'mean_orientation': np.mean(orientations),
                    'std_orientation': np.std(orientations),
                    'min_orientation': min(orientations),
                    'max_orientation': max(orientations)
                }
        stats['orientation_summary'] = orientation_summary
        # Remove raw orientation data
        del stats['orientation_stats']
        
        return stats

def main():
    """Main function with configuration"""
    # Configuration
    YOLO_MODEL = "models/best_yolo.pt"
    CNN_MODEL = "models/agent_cnn.pth"
    ORIENTATION_MODEL = "models/orien_cnn.pth"
    VIDEO_PATH = "input_files/example_video.mp4"
    OUTPUT_VIDEO = "track_output/tracked_vid.mp4"
    OUTPUT_JSON = "track_output/tracked_data.json"
    
    # Initialize tracker with agent grouping enabled (set to False to disable)
    tracker = ValorantVideoTracker(
        YOLO_MODEL, CNN_MODEL, ORIENTATION_MODEL, 
        use_agent_grouping=False  # Toggle agent group manager on/off
    )
    
    print("Valorant Video Tracker")
    print()
    
    # Process video
    tracking_data = tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_VIDEO,
        output_json=OUTPUT_JSON,
        conf_threshold=0.7,
        show_preview=False,  # Set to True for live preview
        save_frequency=0,
        fps_override=None  # Set to None to use detected FPS, or specify (e.g., 30, 60)
    )
    
    #  analysis
    stats = tracker.analyze_tracking_data(tracking_data)
    
    print("\nTracking Statistics:")
    print(f"Total unique agent tracks: {stats['total_agent_tracks']}")
    print(f"Agents per team: {stats['agents_per_team']}")
    print(f"Average track length: {stats['average_track_length']:.1f} frames")
    print(f"Prediction rate: {stats['prediction_stats']['prediction_rate']:.2%}")
    print(f"Total predictions made: {stats['prediction_stats']['total_predictions']}")
    
    print(f"\nOrientation Statistics:")
    for agent, orient_stats in stats['orientation_summary'].items():
        print(f"  {agent}:")
        print(f"    Mean orientation: {orient_stats['mean_orientation']:.1f}°")
        print(f"    Std deviation: {orient_stats['std_orientation']:.1f}°")
    
    print(f"\nUtility counts:")
    for util_type, count in stats['utility_counts'].items():
        print(f"  {util_type}: {count}")
    
    # Save statistics
    with open("tracking_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nOutput files saved:")
    print(f"- Output video: {OUTPUT_VIDEO}")
    print(f"- Tracking data: {OUTPUT_JSON}")
    print(f"- Statistics: tracking_stats.json")

if __name__ == "__main__":
    main()