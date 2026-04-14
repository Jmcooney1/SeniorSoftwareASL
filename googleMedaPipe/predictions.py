import numpy as np
import time
import os
from collections import deque

class OneEuroFilter:
    """Smoothes hand landmarks to reduce camera jitter."""
    def __init__(self, min_cutoff=0.5, beta=0.1, d_cutoff=1.0):
        # min_cutoff lowered to 0.5 for smoother path tracking
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
        
        # 69 Features: 63 (hand) + 3 (spatial) + 3 (velocity)
        self.hands_data = {
            'left': {
                'buffer': deque(maxlen=window_size), 
                'filters': [OneEuroFilter() for _ in range(69)], 
                'last_time': 0
            },
            'right': {
                'buffer': deque(maxlen=window_size), 
                'filters': [OneEuroFilter() for _ in range(69)], 
                'last_time': 0
            }
        }
        
        # --- ANTI-JITTER SYSTEM ---
        self.prediction_history = deque(maxlen=15) # Store results of last 15 frames
        self.current_stable_word = "..."
        
        self.prediction_cooldown = 0.35
        self.min_dist_threshold = 85.0 # Loosened for motion path slop

    def load_library(self, path):
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True).item()
                print(f"✅ Engine: Library Loaded ({len(data)} entries).")
                return data
            except:
                return {}
        return {}

    def process_frame(self, hand_landmarks, side_label, nose_lm=None):
        side = side_label.lower()
        if side not in self.hands_data: return "...", 0

        # 1. Feature Extraction
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = raw_pts[0]
        hand_shape = ((raw_pts - wrist) * 10).flatten()
        
        spatial_vec = np.array([wrist[0] - nose_lm.x, wrist[1] - nose_lm.y, 0]) * 10 if nose_lm else np.zeros(3)

        if len(self.hands_data[side]['buffer']) > 0:
            prev_frame = self.hands_data[side]['buffer'][-1]
            velocity_vec = (spatial_vec - prev_frame[63:66]) * 5 
        else:
            velocity_vec = np.zeros(3)

        combined_features = np.concatenate([hand_shape, spatial_vec, velocity_vec])
        filtered_data = np.array([self.hands_data[side]['filters'][i].filter(combined_features[i]) for i in range(69)])
        self.hands_data[side]['buffer'].append(filtered_data)

        # 2. Analyze Current Frame
        raw_word, raw_dist = self._analyze_window(side)
        
        # 3. Apply Hysteresis (Voting)
        self.prediction_history.append(raw_word)
        
        counts = {}
        for w in self.prediction_history:
            counts[w] = counts.get(w, 0) + 1
        
        most_freq_word = max(counts, key=counts.get)
        
        # Logic: Switch to a word only if it appears in at least 5/15 frames
        if counts[most_freq_word] >= 5 and most_freq_word != "...":
            self.current_stable_word = most_freq_word
        # Logic: Only go back to READY if '...' is overwhelmingly present
        elif counts.get("...", 0) > 12:
            self.current_stable_word = "..."

        # Confidence Calculation
        ratio = max(0, min(1, (1 - (raw_dist / self.min_dist_threshold))))
        confidence = int((ratio ** 2) * 100)
        
        return self.current_stable_word, confidence

    def _analyze_window(self, side):
        current_pose = self.hands_data[side]['buffer'][-1]
        best_label, lowest_dist = "...", 999.0
        side_lower, user_target = side.lower(), self.current_user.strip().lower().replace(" ", "_")

        for full_label, lib_seq in self.library.items():
            label_lower = full_label.lower()
            parts = label_lower.split('_')
            
            # Identity Filter
            if len(parts) >= 4 and parts[0] != user_target:
                continue
            
            # Side Filter
            if side_lower not in label_lower and "dual" not in label_lower:
                continue

            try:
                if isinstance(lib_seq, (np.ndarray, list)):
                    last_frame = lib_seq[-1]
                    lib_pose = last_frame.get(side_lower) if isinstance(last_frame, dict) else last_frame
                else: continue

                if lib_pose is None: continue
                
                v1, v2 = current_pose.flatten(), lib_pose.flatten()

                # Multi-Tier Compatibility
                if len(v2) == 63: v2 = np.concatenate([v2, [0,0,0, 0,0,0]])
                elif len(v2) == 66: v2 = np.concatenate([v2, [0,0,0]])
                
                if len(v1) != len(v2): continue
                dist = np.linalg.norm(v1 - v2)

                if dist < lowest_dist:
                    lowest_dist = dist
                    best_label = parts[1] if len(parts) >= 4 else parts[0]
            except: continue

        return (best_label if lowest_dist < self.min_dist_threshold else "..."), lowest_dist

    def reset_hand(self, side):
        s = side.lower()
        if s in self.hands_data:
            self.hands_data[s]['buffer'].clear()
            for f in self.hands_data[s]['filters']: f.x_prev = f.dx_prev = f.t_prev = None