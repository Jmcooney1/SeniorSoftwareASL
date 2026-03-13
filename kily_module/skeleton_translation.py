import sys
import os
import re
import shutil
import cv2
import numpy as np

SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
GOOGLE_MEDIA_PIPE_DIR = os.path.join(SCRIPT_DIR, "googleMedaPipe")
sys.path.insert(0, GOOGLE_MEDIA_PIPE_DIR)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QImage, QPixmap

from kily_module.database import DataBase
from kily_module.SkeletonExtractor import SkeletonExtractor
import kily_module.projectPoints as projectPoints
import mediapipe as mp

# ── Paths — read from shared config.json at project root ───────────────────
def _load_paths():
    # Walk up from this file to find config.json (works from any subfolder depth)
    import json
    search = os.path.abspath(SCRIPT_DIR)
    for _ in range(4):
        candidate = os.path.join(search, "config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                cfg = json.load(f)
            return (
                cfg.get("dataset_path", ""),
                cfg.get("save_dir", os.path.join(SCRIPT_DIR, "savedVideoPoints"))
            )
        search = os.path.dirname(search)
    # Fallback if no config found
    return (
        os.path.join(SCRIPT_DIR, "dataSet", "wlasl-complete"),
        os.path.join(SCRIPT_DIR, "savedVideoPoints")
    )

DB_PATH, SAVE_DIR = _load_paths()
VIDEO_FOLDER      = os.path.join(DB_PATH, "videos")
VIDEO_INDEX       = os.path.join(DB_PATH, "wlasl_class_list.txt")


# ── Helpers ────────────────────────────────────────────────────────────────
def numpy_to_pixmap(frame_bgr, w, h):
    """Convert an OpenCV BGR frame to a QPixmap scaled to (w, h)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h))
    img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ── Worker Thread ──────────────────────────────────────────────────────────
class WorkerThread(QThread):
    log_signal        = pyqtSignal(str)
    done_signal       = pyqtSignal()
    # emits (raw_bgr_frame, skeleton_bgr_frame, word_label)
    frame_signal      = pyqtSignal(object, object, str)

    def __init__(self, sentence: str, db: DataBase):
        super().__init__()
        self.sentence = sentence
        self.db       = db

    # ── internal: extract skeleton AND emit frames for live preview ──
    def _extract_with_preview(self, word: str, video_path: str):
        mp_drawing        = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands_mod      = mp.solutions.hands
        mp_pose_mod       = mp.solutions.pose

        word_dir = os.path.join(SAVE_DIR, word)
        os.makedirs(os.path.join(word_dir, "pose"),     exist_ok=True)
        os.makedirs(os.path.join(word_dir, "hands"),    exist_ok=True)
        os.makedirs(os.path.join(word_dir, "combined"), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_signal.emit(f"❌ Could not open video: {video_path}")
            return

        hands = mp_hands_mod.Hands(
            model_complexity=0, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        pose = mp_pose_mod.Pose(
            model_complexity=2, enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        import csv
        POSE_LMS = list(range(0, 17)) + [23, 24]

        hand_csv     = open(os.path.join(word_dir, "hands",    "hands_output.csv"),    "w", newline="")
        pose_csv     = open(os.path.join(word_dir, "pose",     "pose_output.csv"),     "w", newline="")
        combined_csv = open(os.path.join(word_dir, "combined", "combined_output.csv"), "w", newline="")

        hw = csv.writer(hand_csv)
        pw = csv.writer(pose_csv)
        cw = csv.writer(combined_csv)

        hw.writerow(["frame", "hand_index", "landmark_index", "x", "y", "z"])
        pw.writerow(["frame", "landmark_index", "x", "y", "z"])
        cw.writerow(["frame",
                     "hand_index", "hand_landmark_index", "hand_x", "hand_y", "hand_z",
                     "pose_landmark_index", "pose_x", "pose_y", "pose_z"])

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)

            # ── Raw frame with MediaPipe overlay ──
            raw_display = frame.copy()
            if pose_results and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    raw_display,
                    pose_results.pose_landmarks,
                    mp_pose_mod.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if hand_results and hand_results.multi_hand_landmarks:
                for hl in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        raw_display, hl,
                        mp_hands_mod.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )

            # ── Skeleton-only frame ──
            h, w = frame.shape[:2]
            skel = np.zeros((h, w, 3), dtype=np.uint8)
            if pose_results and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    skel,
                    pose_results.pose_landmarks,
                    mp_pose_mod.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if hand_results and hand_results.multi_hand_landmarks:
                for hl in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        skel, hl,
                        mp_hands_mod.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )

            self.frame_signal.emit(raw_display.copy(), skel.copy(), word)

            # ── Save CSVs ──
            if hand_results and hand_results.multi_hand_landmarks:
                for hi, hl in enumerate(hand_results.multi_hand_landmarks):
                    for li, lm in enumerate(hl.landmark):
                        hw.writerow([frame_idx, hi, li, lm.x, lm.y, lm.z])
                        cw.writerow([frame_idx, hi, li, lm.x, lm.y, lm.z,
                                     None, None, None, None])

            if pose_results and pose_results.pose_landmarks:
                for li in POSE_LMS:
                    lm = pose_results.pose_landmarks.landmark[li]
                    pw.writerow([frame_idx, li, lm.x, lm.y, lm.z])
                    cw.writerow([frame_idx, None, None, None, None, None,
                                 li, lm.x, lm.y, lm.z])

            frame_idx += 1

        cap.release()
        hand_csv.close()
        pose_csv.close()
        combined_csv.close()
        hands.close()
        pose.close()

    def run(self):
        # Clear previous output
        if os.path.exists(SAVE_DIR):
            for item in os.listdir(SAVE_DIR):
                p = os.path.join(SAVE_DIR, item)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

        words = [w for w in re.split(r'[;,\s]+', self.sentence.strip()) if w]

        # Phase 1: extract skeleton + live preview
        for word in words:
            video_path = self.db.get_video_path(word)
            if video_path is None or "Warning" in str(video_path):
                self.log_signal.emit(f"⚠️  '{word}' — not in database, skipping.")
                continue
            self.log_signal.emit(f"▶  Extracting: {word}")
            self._extract_with_preview(word, video_path)
            self.log_signal.emit(f"✅  Done extracting: {word}")

        # Phase 2: project points
        for word in words:
            word_dir = os.path.join(SAVE_DIR, word)
            if os.path.isdir(word_dir):
                self.log_signal.emit(f"🎞  Projecting: {word}")
                try:
                    projectPoints.run(word_dir)
                except Exception as e:
                    self.log_signal.emit(f"❌  Projection error '{word}': {e}")

        self.done_signal.emit()


# ── Main Window ────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Skeleton Translator")
        self.setMinimumSize(1200, 750)
        self.db = None
        self._build_ui()
        QTimer.singleShot(100, self._load_database)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # Title
        title = QLabel("ASL Skeleton Translator")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:bold; padding:4px;")
        root.addWidget(title)

        self.db_status = QLabel("⏳ Loading database…")
        self.db_status.setStyleSheet("color:gray; font-size:12px;")
        root.addWidget(self.db_status)

        # Input row
        row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a word or sentence to translate…")
        self.input_field.setStyleSheet("padding:8px; font-size:14px;")
        self.input_field.textChanged.connect(self._on_input_changed)
        self.input_field.returnPressed.connect(self.start_processing)

        self.run_btn = QPushButton("▶  Translate")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton          { padding:8px 20px; font-size:14px;
                                   background:#2563eb; color:white; border-radius:6px; }
            QPushButton:disabled { background:#94a3b8; }
            QPushButton:hover    { background:#1d4ed8; }
        """)
        self.run_btn.clicked.connect(self.start_processing)
        row.addWidget(self.input_field)
        row.addWidget(self.run_btn)
        root.addLayout(row)

        # ── Tabs ──
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { padding:6px 16px; font-size:13px; }")

        # Tab 1: Live Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        self.word_label = QLabel("Waiting…")
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setStyleSheet("font-size:16px; font-weight:bold; padding:4px;")
        preview_layout.addWidget(self.word_label)

        video_row = QHBoxLayout()

        # Raw + mediapipe
        left_col = QVBoxLayout()
        lbl_raw = QLabel("📷  Camera / Video + MediaPipe")
        lbl_raw.setStyleSheet("font-weight:bold; font-size:12px;")
        lbl_raw.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_label = QLabel()
        self.raw_label.setFixedSize(480, 320)
        self.raw_label.setStyleSheet("background:black; border:1px solid #333;")
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col.addWidget(lbl_raw)
        left_col.addWidget(self.raw_label)
        video_row.addLayout(left_col)

        # Skeleton only
        right_col = QVBoxLayout()
        lbl_skel = QLabel("🦴  Skeleton Projection")
        lbl_skel.setStyleSheet("font-weight:bold; font-size:12px;")
        lbl_skel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.skel_label = QLabel()
        self.skel_label.setFixedSize(480, 320)
        self.skel_label.setStyleSheet("background:black; border:1px solid #333;")
        self.skel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_col.addWidget(lbl_skel)
        right_col.addWidget(self.skel_label)
        video_row.addLayout(right_col)

        preview_layout.addLayout(video_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(130)
        self.log_box.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:monospace; font-size:12px; border-radius:4px;"
        )
        preview_layout.addWidget(self.log_box)
        tabs.addTab(preview_widget, "🎬  Live Preview")

        # Tab 2: Database table
        db_widget = QWidget()
        db_layout = QVBoxLayout(db_widget)
        db_layout.setContentsMargins(8, 8, 8, 8)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search words…")
        self.search_field.setStyleSheet("padding:6px; font-size:13px;")
        self.search_field.textChanged.connect(self._filter_table)
        db_layout.addWidget(self.search_field)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Word", "Video Index", "Status"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget{font-size:13px;} QHeaderView::section{font-weight:bold; padding:4px;}"
        )
        db_layout.addWidget(self.table)
        tabs.addTab(db_widget, "📖  Database")

        root.addWidget(tabs)

    # ── Database ───────────────────────────────────────────────────────────
    def _load_database(self):
        missing = [p for p in [DB_PATH, VIDEO_FOLDER, VIDEO_INDEX] if not os.path.exists(p)]
        if missing:
            self.db_status.setText("❌ Path not found — check DB_PATH in main.py")
            self.db_status.setStyleSheet("color:red; font-size:12px;")
            self.log("Missing:\n" + "\n".join(missing))
            return
        try:
            self.db = DataBase(database_path=DB_PATH, video_folder=VIDEO_FOLDER)
            self.db.build_dictionary(word_to_video_path=VIDEO_INDEX, video_folder_path=VIDEO_FOLDER)
            self._populate_table()
            n = len(self.db.word_to_path)
            self.db_status.setText(f"✅ Database loaded — {n} words available")
            self.db_status.setStyleSheet("color:green; font-size:12px;")
            self.run_btn.setEnabled(True)
            self.log(f"Database ready. {n} words loaded.")
        except Exception as e:
            self.db_status.setText(f"❌ {e}")
            self.db_status.setStyleSheet("color:red; font-size:12px;")

    def _populate_table(self):
        words = sorted(self.db.word_to_path.keys())
        self.table.setRowCount(len(words))
        for row, word in enumerate(words):
            self.table.setItem(row, 0, QTableWidgetItem(word))
            self.table.setItem(row, 1, QTableWidgetItem(self.db.word_to_path[word]))
            s = QTableWidgetItem("✅ Available")
            s.setForeground(QColor("#16a34a"))
            self.table.setItem(row, 2, s)

    def _filter_table(self, text):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            self.table.setRowHidden(row, text.lower() not in (item.text().lower() if item else ""))

    def _on_input_changed(self, text):
        if not self.db:
            return
        typed = {w.lower() for w in re.split(r'[;,\s]+', text.strip()) if w}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not item:
                continue
            hit = item.text().lower() in typed
            bg  = QColor("#fef08a") if hit else QColor("transparent")
            for col in range(3):
                cell = self.table.item(row, col)
                if cell:
                    cell.setBackground(bg)

    # ── Processing ─────────────────────────────────────────────────────────
    def start_processing(self):
        sentence = self.input_field.text().strip()
        if not sentence or not self.db:
            return
        self.run_btn.setEnabled(False)
        self.log_box.clear()
        self.log(f'🔤 Translating: "{sentence}"')

        self.thread = WorkerThread(sentence, self.db)
        self.thread.log_signal.connect(self.log)
        self.thread.frame_signal.connect(self._update_preview)
        self.thread.done_signal.connect(self.on_done)
        self.thread.start()

    def _update_preview(self, raw_frame, skel_frame, word):
        self.word_label.setText(f"Word: {word}")
        self.raw_label.setPixmap(numpy_to_pixmap(raw_frame, 480, 320))
        self.skel_label.setPixmap(numpy_to_pixmap(skel_frame, 480, 320))

    def log(self, msg):
        self.log_box.append(msg)

    def on_done(self):
        self.log("✅ All done!")
        self.run_btn.setEnabled(True)
        self.word_label.setText("✅ Complete")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())