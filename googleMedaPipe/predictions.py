import numpy as np
import time
import os
from collections import deque

class OneEuroFilter:
    """Smoothes hand landmarks to reduce camera jitter."""
    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

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
        res = self._low_pass(x, self.x_prev, alpha)
        self.t_prev, self.x_prev, self.dx_prev = t, res, edx
        return res

class MotionPredictor:
    def __init__(self, library_file='asl_motion_library.npy', window_size=28):
        self.window_size = window_size
        self.library = self.load_library(library_file)
        self.current_user = "default" 
        
        self.hands_data = {
            'left': {'buffer': deque(maxlen=window_size), 'filters': [OneEuroFilter() for _ in range(66)], 'last_time': 0},
            'right': {'buffer': deque(maxlen=window_size), 'filters': [OneEuroFilter() for _ in range(66)], 'last_time': 0}
        }
        
        self.prediction_cooldown = 0.35
        self.min_dist_threshold = 60.0 # Slightly loosened for easier matching

    def load_library(self, path):
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True).item()
                return data
            except: return {}
        return {}

    def process_frame(self, hand_landmarks, side_label, nose_lm=None):
        side = side_label.lower()
        if side not in self.hands_data: return "...", 0

        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = raw_pts[0]
        hand_shape = ((raw_pts - wrist) * 10).flatten()
        spatial_vec = np.array([wrist[0] - nose_lm.x, wrist[1] - nose_lm.y, 0]) * 10 if nose_lm else np.zeros(3)
        combined_features = np.concatenate([hand_shape, spatial_vec])

        filtered_data = np.array([self.hands_data[side]['filters'][i].filter(combined_features[i]) for i in range(66)])
        self.hands_data[side]['buffer'].append(filtered_data)

        current_time = time.time()
        hand = self.hands_data[side]
        
        if len(hand['buffer']) >= 10:
            word, raw_dist = self._analyze_window(side)
            ratio = max(0, min(1, (1 - (raw_dist / self.min_dist_threshold))))
            confidence = int((ratio ** 2) * 100)
            
            if word != "..." and (current_time - hand['last_time']) > self.prediction_cooldown:
                if confidence > 55: # Lowered confidence threshold slightly for live testing
                    hand['last_time'] = current_time
                    return word, confidence
            return "...", confidence
        return "...", 0

    def _analyze_window(self, side):
        current_pose = self.hands_data[side]['buffer'][-1]
        best_label, lowest_dist = "...", 999.0
        side_lower = side.lower()
        
        # Normalize target user
        user_target = self.current_user.strip().lower().replace(" ", "_")

        for full_label, lib_seq in self.library.items():
            label_lower = full_label.lower()
            parts = label_lower.split('_')
            
            # --- IMPROVED IDENTITY LOGIC ---
            # If the file belongs to SOMEONE ELSE, skip it.
            # A file belongs to someone else if it has 4+ parts and part[0] != user_target
            if len(parts) >= 4 and parts[0] != user_target:
                continue
            
            # Side check
            if side_lower not in label_lower and "dual" not in label_lower:
                continue

            try:
                if isinstance(lib_seq, (np.ndarray, list)):
                    last_frame = lib_seq[-1]
                    lib_pose = last_frame.get(side_lower) if isinstance(last_frame, dict) else last_frame
                else: continue

                if lib_pose is None: continue
                
                v1 = current_pose.flatten()
                v2 = lib_pose.flatten()

                # Backward Compatibility (63 -> 66)
                if len(v2) == 63: v2 = np.concatenate([v2, [0, 0, 0]])
                if len(v1) != len(v2): continue
                
                dist = np.linalg.norm(v1 - v2)

                if dist < lowest_dist:
                    lowest_dist = dist
                    # Format A (User): user_word_side_index -> parts[1]
                    # Format B (Legacy): word_side_index -> parts[0]
                    best_label = parts[1] if len(parts) >= 4 else parts[0]
            except: continue

        final_label = best_label if lowest_dist < self.min_dist_threshold else "..."
        return final_label, lowest_dist

    def reset_hand(self, side):
        s = side.lower()
        if s in self.hands_data:
            self.hands_data[s]['buffer'].clear()
            for f in self.hands_data[s]['filters']: f.x_prev = f.dx_prev = f.t_prev = None