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

    def _get_alpha(self, te, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

class MotionPredictor:
    def __init__(self, library_file='asl_motion_library.npy', window_size=28):
        self.window_size = window_size
        self.library = self.load_library(library_file)
        
        self.hands_data = {
            'left': {
                'buffer': deque(maxlen=window_size), 
                'filters': [OneEuroFilter() for _ in range(63)], 
                'last_time': 0
            },
            'right': {
                'buffer': deque(maxlen=window_size), 
                'filters': [OneEuroFilter() for _ in range(63)], 
                'last_time': 0
            }
        }
        
        self.prediction_cooldown = 0.35
        # Increased threshold slightly to prevent "flickering" off/on
        self.min_dist_threshold = 30.0 

    def load_library(self, path):
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True).item()
                print(f"✅ Library Loaded: {len(data)} entries.")
                return data
            except: return {}
        return {}

    def process_frame(self, hand_landmarks, side_label):
        """
        Returns (word, confidence_percentage)
        """
        side = side_label.lower()
        if side not in self.hands_data: 
            return "...", 0

        ## ... (keep normalization and filtering exactly the same) ...
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = raw_pts[0]
        scale = np.linalg.norm(raw_pts[0] - raw_pts[9]) or 1.0
        normalized = ((raw_pts - wrist) / scale).flatten()

        filtered_data = np.array([
            self.hands_data[side]['filters'][i].filter(normalized[i]) 
            for i in range(len(normalized))
        ])
        self.hands_data[side]['buffer'].append(filtered_data)

        current_time = time.time()
        hand = self.hands_data[side]
        
        if len(hand['buffer']) >= 10:
            word, raw_dist = self._analyze_window(side)
            
            # --- TOUGHER GRADING LOGIC ---
            # We calculate a 'linear' ratio first (0.0 to 1.0)
            ratio = max(0, min(1, (1 - (raw_dist / self.min_dist_threshold))))
            
            # Applying a Power Curve (ratio^2 or ratio^3) makes the high scores harder to get.
            # Example: A ratio of 0.9 (which was 90%) now becomes 0.9 * 0.9 = 0.81 (81%)
            # You have to be EXTREMELY close to get that 95%+.
            confidence = int((ratio ** 2) * 100)
            
            # We trigger the word if the confidence is high enough (e.g., > 60%)
            if word != "..." and (current_time - hand['last_time']) > self.prediction_cooldown:
                if confidence > 60: 
                    hand['last_time'] = current_time
                    return word, confidence
            
            return "...", confidence
        
        return "...", 0

    def _analyze_window(self, side):
        """Finds the closest match. Returns (best_label, distance)"""
        current_pose = self.hands_data[side]['buffer'][-1]
        best_label = "..."
        lowest_dist = 999.0 

        for full_label, lib_seq in self.library.items():
            try:
                if hasattr(lib_seq, 'shape') and len(lib_seq.shape) > 1:
                    lib_pose = lib_seq[-1]
                else:
                    lib_pose = lib_seq
                
                if len(current_pose) != len(lib_pose.flatten()): 
                    continue
                
                dist = np.linalg.norm(current_pose - lib_pose.flatten())

                if dist < lowest_dist:
                    lowest_dist = dist
                    parts = full_label.split('_')
                    if len(parts) > 1:
                        best_label = parts[1] if parts[0].lower() in ['left', 'right'] else parts[0]
                    else:
                        best_label = full_label
            except:
                continue

        # If the distance is above our threshold, we don't return the word,
        # but we ALWAYS return the lowest_dist so the meter can show it.
        final_label = best_label if lowest_dist < self.min_dist_threshold else "..."
        return final_label, lowest_dist

    def reset_hand(self, side):
        s = side.lower()
        if s in self.hands_data:
            self.hands_data[s]['buffer'].clear()
            for f in self.hands_data[s]['filters']:
                f.x_prev = f.dx_prev = f.t_prev = None