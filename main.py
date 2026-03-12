import sys
import os
import re
import shutil
import time
import cv2
import numpy as np
import csv
import traceback

SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
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

from database import DataBase
import mediapipe as mp

# ── Fix MediaPipe resource path when bundled with PyInstaller ──────────────
def _fix_mediapipe_path():
    """
    When frozen by PyInstaller, MediaPipe can't find its model files because
    it looks relative to the original install location. Point it at the
    bundled copy inside the .app's Frameworks directory.
    """
    import importlib.resources
    import mediapipe as _mp

    if getattr(sys, "frozen", False):
        # We're running inside a PyInstaller bundle
        bundle_dir = os.path.dirname(sys.executable)          # .../MacOS/
        frameworks = os.path.join(bundle_dir, "..", "Frameworks")
        mp_data    = os.path.normpath(os.path.join(frameworks, "mediapipe"))
        if os.path.isdir(mp_data):
            os.environ["MEDIAPIPE_RESOURCE_DIR"] = mp_data
            # Also patch the module-level path that solution_base.py uses
            try:
                import mediapipe.python.solution_base as _sb
                _sb._resource_dir = mp_data
            except Exception:
                pass

_fix_mediapipe_path()

# ── Paths ──────────────────────────────────────────────────────────────────
def _find_dataset():
    candidates = [
        os.path.join(SCRIPT_DIR, "dataSet", "wlasl-complete"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "dataSet", "wlasl-complete"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "dataSet", "wlasl-complete"),
        os.path.join(os.path.expanduser("~"), "Desktop", "SeniorSoftwareASL", "dataSet", "wlasl-complete"),
        os.path.join(os.path.expanduser("~"), "Desktop", "dataSet", "wlasl-complete"),
        os.path.join(os.getcwd(), "dataSet", "wlasl-complete"),
    ]
    for path in candidates:
        resolved = os.path.normpath(path)
        if os.path.isdir(resolved):
            return resolved
    return os.path.normpath(candidates[3])

DB_PATH      = _find_dataset()
VIDEO_FOLDER = os.path.join(DB_PATH, "videos")
VIDEO_INDEX  = os.path.join(DB_PATH, "wlasl_class_list.txt")
SAVE_DIR     = os.path.join(SCRIPT_DIR, "googleMedaPipe", "savedVideoPoints")

PREVIEW_W    = 480
PREVIEW_H    = 320
TARGET_FPS   = 30


# ── Frame conversion ───────────────────────────────────────────────────────
def bgr_to_pixmap(frame_bgr, w, h):
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb   = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    img   = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ── Worker Thread ──────────────────────────────────────────────────────────
class WorkerThread(QThread):
    log_signal   = pyqtSignal(str)
    done_signal  = pyqtSignal()
    frame_signal = pyqtSignal(np.ndarray, np.ndarray, str)

    def __init__(self, sentence: str, db: DataBase):
        super().__init__()
        self.sentence = sentence
        self.db       = db

    def _extract(self, word: str, video_path: str):
        # Re-apply mediapipe path fix in the worker thread context
        _fix_mediapipe_path()

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
            self.log_signal.emit(f"❌ Cannot open: {video_path}")
            return

        video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / min(TARGET_FPS, video_fps)
        POSE_LMS    = list(range(0, 17)) + [23, 24]

        hands_mp = mp_hands_mod.Hands(
            model_complexity=0, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        pose_mp = mp_pose_mod.Pose(
            model_complexity=2, enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        hf = open(os.path.join(word_dir, "hands",    "hands_output.csv"),    "w", newline="")
        pf = open(os.path.join(word_dir, "pose",     "pose_output.csv"),     "w", newline="")
        cf = open(os.path.join(word_dir, "combined", "combined_output.csv"), "w", newline="")
        hw = csv.writer(hf); pw = csv.writer(pf); cw = csv.writer(cf)
        hw.writerow(["frame","hand_index","landmark_index","x","y","z"])
        pw.writerow(["frame","landmark_index","x","y","z"])
        cw.writerow(["frame",
                     "hand_index","hand_landmark_index","hand_x","hand_y","hand_z",
                     "pose_landmark_index","pose_x","pose_y","pose_z"])

        frame_idx = 0
        last_emit = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = hands_mp.process(rgb)
                pose_results = pose_mp.process(rgb)

                raw = frame.copy()
                if pose_results and pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        raw, pose_results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                if hand_results and hand_results.multi_hand_landmarks:
                    for hl in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            raw, hl, mp_hands_mod.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                        )

                h, w = frame.shape[:2]
                skel = np.zeros((h, w, 3), dtype=np.uint8)
                if pose_results and pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        skel, pose_results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                if hand_results and hand_results.multi_hand_landmarks:
                    for hl in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            skel, hl, mp_hands_mod.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                        )

                now = time.monotonic()
                if now - last_emit >= frame_delay:
                    self.frame_signal.emit(raw.copy(), skel.copy(), word)
                    last_emit = now

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

        finally:
            cap.release()
            hf.close(); pf.close(); cf.close()
            hands_mp.close(); pose_mp.close()

    # ── Main run loop — fully wrapped so PyQt6 never sees an unhandled exception ──
    def run(self):
        try:
            if os.path.exists(SAVE_DIR):
                for item in os.listdir(SAVE_DIR):
                    p = os.path.join(SAVE_DIR, item)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

            words = [w for w in re.split(r'[;,\s]+', self.sentence.strip()) if w]

            # Phase 1 — extract + live preview
            for word in words:
                try:
                    video_path = self.db.get_video_path(word)
                    if video_path is None or "Warning" in str(video_path):
                        self.log_signal.emit(f"⚠️  '{word}' — not in database, skipping.")
                        continue
                    self.log_signal.emit(f"▶  Extracting: {word}")
                    self._extract(word, video_path)
                    self.log_signal.emit(f"✅  Saved: {word}")
                except Exception as e:
                    self.log_signal.emit(f"❌  Error on '{word}': {e}")
                    self.log_signal.emit(traceback.format_exc())

        except Exception as e:
            self.log_signal.emit(f"❌  Fatal worker error: {e}")
            self.log_signal.emit(traceback.format_exc())

        finally:
            self.done_signal.emit()


# ── Main Window ────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Skeleton Translator")
        self.setMinimumSize(1200, 780)
        self.db = None
        self._build_ui()
        QTimer.singleShot(100, self._load_database)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        title = QLabel("ASL Skeleton Translator")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:bold; padding:4px;")
        root.addWidget(title)

        self.db_status = QLabel("⏳ Loading database…")
        self.db_status.setStyleSheet("color:gray; font-size:12px;")
        root.addWidget(self.db_status)

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

        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { padding:6px 18px; font-size:13px; }")

        # ── Tab 1: Live Preview ──
        preview_widget = QWidget()
        pv = QVBoxLayout(preview_widget)
        pv.setContentsMargins(8, 8, 8, 8)
        pv.setSpacing(8)

        self.word_label = QLabel("Waiting for input…")
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setStyleSheet(
            "font-size:17px; font-weight:bold; padding:4px; color:#1d4ed8;"
        )
        pv.addWidget(self.word_label)

        video_row = QHBoxLayout()
        video_row.setSpacing(16)

        for attr, heading in [("raw_label",  "📷  Video + MediaPipe Landmarks"),
                               ("skel_label", "🦴  Skeleton Projection")]:
            col = QVBoxLayout()
            col.setSpacing(4)
            h = QLabel(heading)
            h.setAlignment(Qt.AlignmentFlag.AlignCenter)
            h.setStyleSheet("font-weight:bold; font-size:12px; color:#374151;")
            lbl = QLabel()
            lbl.setFixedSize(PREVIEW_W, PREVIEW_H)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setText("No signal")
            lbl.setStyleSheet(
                "background:#111; border:2px solid #374151; border-radius:6px;"
                "color:#555; font-size:14px;"
            )
            setattr(self, attr, lbl)
            col.addWidget(h)
            col.addWidget(lbl)
            video_row.addLayout(col)

        pv.addLayout(video_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        self.log_box.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:monospace;"
            "font-size:12px; border-radius:4px;"
        )
        pv.addWidget(self.log_box)
        tabs.addTab(preview_widget, "🎬  Live Preview")

        # ── Tab 2: Database ──
        db_widget = QWidget()
        dv = QVBoxLayout(db_widget)
        dv.setContentsMargins(8, 8, 8, 8)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search words…")
        self.search_field.setStyleSheet("padding:6px; font-size:13px;")
        self.search_field.textChanged.connect(self._filter_table)
        dv.addWidget(self.search_field)

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
        dv.addWidget(self.table)
        tabs.addTab(db_widget, "📖  Database")

        root.addWidget(tabs)

    # ── Database ───────────────────────────────────────────────────────────
    def _load_database(self):
        bad = [p for p in [DB_PATH, VIDEO_FOLDER, VIDEO_INDEX] if not os.path.exists(p)]
        if bad:
            self.db_status.setText("❌ Dataset not found — see log for details")
            self.db_status.setStyleSheet("color:red; font-size:12px;")
            self.log("❌ Could not find dataset. Please place it at:")
            self.log("   ~/Desktop/SeniorSoftwareASL/dataSet/wlasl-complete")
            self.log(f"\nSearched: {DB_PATH}")
            return
        try:
            self.db = DataBase(database_path=DB_PATH, video_folder=VIDEO_FOLDER)
            self.db.build_dictionary(word_to_video_path=VIDEO_INDEX,
                                     video_folder_path=VIDEO_FOLDER)
            self._populate_table()
            n = len(self.db.word_to_path)
            self.db_status.setText(f"✅ Database loaded — {n} words available")
            self.db_status.setStyleSheet("color:green; font-size:12px;")
            self.run_btn.setEnabled(True)
            self.log(f"Database ready. {n} words loaded.")
        except Exception as e:
            self.db_status.setText(f"❌ {e}")
            self.db_status.setStyleSheet("color:red; font-size:12px;")
            self.log(f"Error loading database: {e}")
            self.log(traceback.format_exc())

    def _populate_table(self):
        words = sorted(self.db.word_to_path.keys())
        self.table.setRowCount(len(words))
        for r, word in enumerate(words):
            self.table.setItem(r, 0, QTableWidgetItem(word))
            self.table.setItem(r, 1, QTableWidgetItem(self.db.word_to_path[word]))
            s = QTableWidgetItem("✅ Available")
            s.setForeground(QColor("#16a34a"))
            self.table.setItem(r, 2, s)

    def _filter_table(self, text):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            self.table.setRowHidden(r, text.lower() not in (item.text().lower() if item else ""))

    def _on_input_changed(self, text):
        if not self.db:
            return
        typed = {w.lower() for w in re.split(r'[;,\s]+', text.strip()) if w}
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if not item:
                continue
            bg = QColor("#fef08a") if item.text().lower() in typed else QColor("transparent")
            for c in range(3):
                cell = self.table.item(r, c)
                if cell:
                    cell.setBackground(bg)

    # ── Processing ─────────────────────────────────────────────────────────
    def start_processing(self):
        sentence = self.input_field.text().strip()
        if not sentence or not self.db:
            return
        self.run_btn.setEnabled(False)
        self.log_box.clear()
        self.raw_label.clear()
        self.skel_label.clear()
        self.word_label.setText("Starting…")
        self.log(f'🔤 Translating: "{sentence}"')

        self.thread = WorkerThread(sentence, self.db)
        self.thread.log_signal.connect(self.log, Qt.ConnectionType.QueuedConnection)
        self.thread.done_signal.connect(self.on_done, Qt.ConnectionType.QueuedConnection)
        self.thread.frame_signal.connect(self._update_preview,
                                         Qt.ConnectionType.QueuedConnection)
        self.thread.start()

    def _update_preview(self, raw: np.ndarray, skel: np.ndarray, word: str):
        self.word_label.setText(f"Word: {word}")
        self.raw_label.setPixmap(bgr_to_pixmap(raw,  PREVIEW_W, PREVIEW_H))
        self.skel_label.setPixmap(bgr_to_pixmap(skel, PREVIEW_W, PREVIEW_H))

    def log(self, msg: str):
        self.log_box.append(msg)

    def on_done(self):
        self.log("✅ All done!")
        self.word_label.setText("✅ Complete")
        self.run_btn.setEnabled(True)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())