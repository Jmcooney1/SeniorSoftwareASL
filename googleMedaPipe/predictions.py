import numpy as np
import time
import os
from collections import deque

class OneEuroFilter:
    """
    Industry-standard smoothing filter. 
    Reduces jitter when still and lag when moving fast.
    """
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
        
        # Multi-hand data tracking
        self.hands_data = {
            'left': {
                'buffer': deque(maxlen=window_size),
                'filters': [OneEuroFilter() for _ in range(63)],
                'last_pred_time': 0
            },
            'right': {
                'buffer': deque(maxlen=window_size),
                'filters': [OneEuroFilter() for _ in range(63)],
                'last_pred_time': 0
            }
        }
        
        self.prediction_cooldown = 0.7  # Seconds between repeats
        self.min_dist_threshold = 12.0 # Adjust based on library quality

    def load_library(self, path):
        if os.path.exists(path):
            print(f"✅ Motion Library Loaded: {path}")
            return np.load(path, allow_pickle=True).item()
        print(f"⚠️ Warning: {path} not found.")
        return {}

    def process_frame(self, hand_landmarks, side_label):
        """
        Processes a single hand from the frame.
        side_label: 'left' or 'right'
        """
        side = side_label.lower()
        if side not in self.hands_data:
            return "..."

        # 1. Extract and Normalize (Wrist at origin)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        normalized = (raw_pts - raw_pts[0]).flatten()

        # 2. Filter using specific filters for THIS hand
        filtered_data = np.array([
            self.hands_data[side]['filters'][i].filter(normalized[i]) 
            for i in range(len(normalized))
        ])
        
        self.hands_data[side]['buffer'].append(filtered_data)

        # 3. Hybrid Prediction Logic (Static & Motion)
        current_time = time.time()
        hand = self.hands_data[side]
        
        # Need at least 15 frames for a decent comparison
        if len(hand['buffer']) >= 15:
            if (current_time - hand['last_pred_time']) > self.prediction_cooldown:
                
                # Check if hand is static (for signs like 'Me')
                is_still = self._is_hand_still(hand['buffer'])
                prediction = self._analyze_window(side, is_still)
                
                if prediction != "...":
                    hand['last_pred_time'] = current_time
                    # Partial clear so you don't have to wait for total refill
                    for _ in range(12): 
                        if hand['buffer']: hand['buffer'].popleft()
                    return prediction
        
        return "..."

    def _is_hand_still(self, buffer):
        """Detects if the hand is holding a pose (low variance)."""
        if len(buffer) < 5: return False
        recent = np.array(list(buffer))[-5:]
        variance = np.var(recent, axis=0).mean()
        return variance < 0.0008

    def _analyze_window(self, side, is_still):
        buffer_list = list(self.hands_data[side]['buffer'])
        current_window = np.array(buffer_list)
        
        best_match = "..."
        # Stricter threshold for static signs, looser for motion
        threshold = 6.0 if is_still else self.min_dist_threshold

        for label, lib_sequence in self.library.items():
            # Check side influence (skip if hand side doesn't match label)
            if side not in label.lower():
                continue

            if is_still:
                # Compare only the latest pose
                v1 = current_window[-1]
                v2 = lib_sequence[-1]
            else:
                # Compare the trajectory 'snake'
                comp_len = min(len(current_window), len(lib_sequence), 20)
                v1 = current_window[-comp_len:].flatten()
                v2 = lib_sequence[-comp_len:].flatten()
            
            dist = np.linalg.norm(v1 - v2)
            
            if dist < threshold:
                threshold = dist
                best_match = label.split('_')[0]

        return best_match

    def reset_hand(self, side):
        """Clears buffers and filters for a specific hand side."""
        side = side.lower()
        if side in self.hands_data:
            self.hands_data[side]['buffer'].clear()
            for f in self.hands_data[side]['filters']:
                f.x_prev = f.dx_prev = f.t_prev = None