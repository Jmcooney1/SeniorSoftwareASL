import numpy as np
import time
import os
from collections import deque

class OneEuroFilter:
    """
    Industry-standard smoothing filter to reduce jitter in hand landmarks.
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
        
        # Multi-hand data tracking
        self.hands_data = {
            'left': {
                'buffer': deque(maxlen=window_size),
                'filters': [OneEuroFilter() for _ in range(63)], # 21 landmarks * 3 (x,y,z)
                'last_pred_time': 0
            },
            'right': {
                'buffer': deque(maxlen=window_size),
                'filters': [OneEuroFilter() for _ in range(63)],
                'last_pred_time': 0
            }
        }
        
        self.prediction_cooldown = 0.6  # Seconds between repeat predictions
        self.min_dist_threshold = 15.0 # Maximum allowed distance for a 'match'

    def load_library(self, path):
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True).item()
                print(f"✅ Motion Library Loaded: {len(data)} entries found.")
                return data
            except Exception as e:
                print(f"❌ Error loading library: {e}")
                return {}
        print(f"⚠️ Warning: {path} not found.")
        return {}

    def process_frame(self, hand_landmarks, side_label):
        """
        Main entry point: Call this every frame for each detected hand.
        """
        side = side_label.lower()
        if side not in self.hands_data:
            return "..."

        # 1. Extract and Normalize (Wrist at 0,0,0)
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        normalized = (raw_pts - raw_pts[0]).flatten()

        # 2. Smooth the data
        filtered_data = np.array([
            self.hands_data[side]['filters'][i].filter(normalized[i]) 
            for i in range(len(normalized))
        ])
        
        self.hands_data[side]['buffer'].append(filtered_data)

        # 3. Prediction Logic
        current_time = time.time()
        hand = self.hands_data[side]
        
        # We need a minimum amount of data to compare trajectories
        if len(hand['buffer']) >= 18:
            if (current_time - hand['last_pred_time']) > self.prediction_cooldown:
                
                is_still = self._is_hand_still(hand['buffer'])
                prediction = self._analyze_window(side, is_still)
                
                if prediction != "...":
                    hand['last_pred_time'] = current_time
                    # Pop some frames so the same motion doesn't trigger twice instantly
                    for _ in range(12): 
                        if hand['buffer']: hand['buffer'].popleft()
                    return prediction
        
        return "..."

    def _is_hand_still(self, buffer):
        """Detects if the hand is holding a pose vs moving."""
        recent = np.array(list(buffer))[-5:]
        variance = np.var(recent, axis=0).mean()
        return variance < 0.0009

    def _analyze_window(self, side, is_still):
        """
        Compares current buffer against ALL library entries.
        Returns the clean label of the mathematically closest match.
        """
        current_window = np.array(list(self.hands_data[side]['buffer']))
        
        best_match_label = "..."
        # Start with the threshold and look for the absolute minimum
        lowest_dist = 6.0 if is_still else self.min_dist_threshold

        for full_label, lib_sequence in self.library.items():
            # Only compare right-hand live data to right-hand recordings
            if side not in full_label.lower():
                continue

            # Distance Comparison
            if is_still:
                # Compare the single latest hand pose
                v1 = current_window[-1]
                v2 = lib_sequence[-1]
            else:
                # Compare the trajectory 'snake' (last 20 frames)
                comp_len = min(len(current_window), len(lib_sequence), 20)
                v1 = current_window[-comp_len:].flatten()
                v2 = lib_sequence[-comp_len:].flatten()
            
            # Euclidean distance (lower is better)
            dist = np.linalg.norm(v1 - v2)
            
            if dist < lowest_dist:
                lowest_dist = dist
                
                # Cleanup: Strips "Right_Ride_3" -> "Ride"
                parts = full_label.split('_')
                if len(parts) >= 2:
                    # If format is Side_Word_Num, index 1 is the word
                    # If format is Word_Num, index 0 is the word
                    best_match_label = parts[1] if side in parts[0].lower() else parts[0]
                else:
                    best_match_label = full_label

        return best_match_label

    def reset_hand(self, side):
        """Call this if a hand leaves the screen."""
        side = side.lower()
        if side in self.hands_data:
            self.hands_data[side]['buffer'].clear()
            for f in self.hands_data[side]['filters']:
                f.x_prev = f.dx_prev = f.t_prev = None