import numpy as np
import time
import os
from collections import deque

class OneEuroFilter:
    """Smoothes hand landmarks to reduce camera jitter."""
    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _low_pass(self, x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev

    def _get_alpha(self, te, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        t = time.time()
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0.0
            return x
        te = t - self.t_prev
        if te <= 0: return x 
        dx = (x - self.x_prev) / te
        edx = self._low_pass(dx, self.dx_prev, self._get_alpha(te, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._get_alpha(te, cutoff)
        result = self._low_pass(x, self.x_prev, alpha)
        self.t_prev, self.x_prev, self.dx_prev = t, result, edx
        return result

class MotionPredictor:
    def __init__(self, library_file='asl_motion_library.npy', window_size=28):
        self.window_size = window_size
        self.library = self.load_library(library_file)
        
        # 66 Filters: 63 for hand joints + 3 for Nose-to-Wrist vector
        self.hands_data = {
            'left': {
                'buffer': deque(maxlen=window_size), 
                'filters': [OneEuroFilter() for _ in range(66)], 
                'last_time': 0
            },
            'right': {
                'buffer': deque(maxlen=window_size), 
                'filters': [OneEuroFilter() for _ in range(66)], 
                'last_time': 0
            }
        }
        
        self.prediction_cooldown = 0.35
        # Threshold for the 66-feature vector
        self.min_dist_threshold = 55.0 

    def load_library(self, path):
        """Loads library or creates a fresh one if missing to prevent boot crashes."""
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True).item()
                print(f"✅ Library Loaded: {len(data)} entries.")
                return data
            except Exception as e:
                print(f"⚠️ Library corrupted, starting fresh: {e}")
                return {}
        else:
            print("📁 No library found. Initializing empty library file...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.save(path, {})
            return {}

    def process_frame(self, hand_landmarks, side_label, nose_lm=None):
        """Main entry point. Maps fingers to wrist, and wrist to nose."""
        side = side_label.lower()
        if side not in self.hands_data: 
            return "...", 0

        # --- 1. INTERNAL HAND SHAPE (Wrist-Centric) ---
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = raw_pts[0]
        hand_shape = ((raw_pts - wrist) * 10).flatten()
        
        # --- 2. GLOBAL SPATIAL POSITION (Nose-Centric) ---
        if nose_lm:
            # Nose Tip from FaceDetection only contains X and Y.
            # Default Z to 0 to maintain 66-length vector.
            spatial_vec = np.array([
                wrist[0] - nose_lm.x, 
                wrist[1] - nose_lm.y, 
                0 
            ]) * 10
        else:
            spatial_vec = np.zeros(3)

        # Combine into 66 features
        combined_features = np.concatenate([hand_shape, spatial_vec])

        # --- 3. SMOOTHING ---
        filtered_data = np.array([
            self.hands_data[side]['filters'][i].filter(combined_features[i]) 
            for i in range(66)
        ])
        self.hands_data[side]['buffer'].append(filtered_data)

        current_time = time.time()
        hand = self.hands_data[side]
        
        # --- 4. PREDICTION & TOUGH SCORING ---
        if len(hand['buffer']) >= 10:
            word, raw_dist = self._analyze_window(side)
            
            # Power Curve Logic: (1 - dist/threshold)^2
            ratio = max(0, min(1, (1 - (raw_dist / self.min_dist_threshold))))
            confidence = int((ratio ** 2) * 100)
            
            if word != "..." and (current_time - hand['last_time']) > self.prediction_cooldown:
                if confidence > 60: 
                    hand['last_time'] = current_time
                    return word, confidence
            
            return "...", confidence
        
        return "...", 0

    def _analyze_window(self, side):
        """Compares current 66-feature signature against the library."""
        current_pose = self.hands_data[side]['buffer'][-1]
        best_label = "..."
        lowest_dist = 999.0 
        side_lower = side.lower()

        for full_label, lib_seq in self.library.items():
            label_lower = full_label.lower()

            if side_lower not in label_lower and "dual" not in label_lower:
                continue

            try:
                if isinstance(lib_seq, (np.ndarray, list)):
                    last_frame = lib_seq[-1]
                    lib_pose = last_frame.get(side_lower) if isinstance(last_frame, dict) else last_frame
                else:
                    continue

                if lib_pose is None or np.all(lib_pose == 0): 
                    continue
                
                v1 = current_pose.flatten()
                v2 = lib_pose.flatten()
                if len(v1) != len(v2): 
                    continue
                
                dist = np.linalg.norm(v1 - v2)

                if dist < lowest_dist:
                    lowest_dist = dist
                    parts = full_label.split('_')
                    if len(parts) > 1:
                        best_label = parts[1] if parts[0].lower() in ['left', 'right', 'dual'] else parts[0]
                    else:
                        best_label = full_label
            except:
                continue

        final_label = best_label if lowest_dist < self.min_dist_threshold else "..."
        return final_label, lowest_dist

    def reset_hand(self, side):
        s = side.lower()
        if s in self.hands_data:
            self.hands_data[s]['buffer'].clear()
            for f in self.hands_data[s]['filters']:
                f.x_prev = f.dx_prev = f.t_prev = None