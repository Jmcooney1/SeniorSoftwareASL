import os
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QThread

# ── Path constants ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # drews_module/
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)                 # project root


def _load_paths():
    import json
    search = os.path.abspath(SCRIPT_DIR)
    for _ in range(4):
        candidate = os.path.join(search, "config.json")
        if os.path.exists(candidate):
            project_root = search
            with open(candidate) as f:
                cfg = json.load(f)
            base = cfg.get("dataset_path", "dataSet")
            save = cfg.get("save_dir", "savedVideoPoints")
            if not os.path.isabs(base):
                base = os.path.join(project_root, base)
            if not os.path.isabs(save):
                save = os.path.join(project_root, save)
            return base, save
        search = os.path.dirname(search)
    project_root = os.path.dirname(SCRIPT_DIR)
    return (
        os.path.join(project_root, "dataSet"),
        os.path.join(project_root, "savedVideoPoints")
    )


# ── Resolve at import time ───────────────────────────────────────────────────
_BASE,   SAVE_DIR = _load_paths()
DB_PATH  = os.path.join(_BASE, "drew-dataset")
DATA_DIR = os.path.join(DB_PATH, "asl_letters")


# ── Background threads — keeps cv2/tkinter off the PyQt6 main thread ────────
class FaceMaskThread(QThread):
    def run(self):
        try:
            from drews_module.faceMask import run
            run()
        except Exception as e:
            print(f"FaceMask error: {e}")


class QuizThread(QThread):
    def run(self):
        try:
            from drews_module.translationquiz import main
            main(data_dir=DATA_DIR)
        except Exception as e:
            print(f"Quiz error: {e}")


# ── Main Window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drew's Module")
        self.setMinimumSize(400, 300)
        self._face_thread = None
        self._quiz_thread = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        title = QLabel("Drew's Module")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e293b;")
        layout.addWidget(title)

        subtitle = QLabel("Select a feature to launch:")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #64748b; font-size: 13px;")
        layout.addWidget(subtitle)

        btn_face = QPushButton("😷  Face Mask / Hand Tracking")
        btn_face.setMinimumHeight(60)
        btn_face.setStyleSheet("""
            QPushButton {
                background: white; color: #1e293b;
                border-radius: 10px; padding: 14px;
                font-size: 14px; border: 2px solid #e2e8f0;
            }
            QPushButton:hover { border-color: #2563eb; background: #eff6ff; }
        """)
        btn_face.clicked.connect(self._launch_facemask)
        layout.addWidget(btn_face)

        btn_quiz = QPushButton("🤟  ASL Translation Quiz")
        btn_quiz.setMinimumHeight(60)
        btn_quiz.setStyleSheet("""
            QPushButton {
                background: white; color: #1e293b;
                border-radius: 10px; padding: 14px;
                font-size: 14px; border: 2px solid #e2e8f0;
            }
            QPushButton:hover { border-color: #2563eb; background: #eff6ff; }
        """)
        btn_quiz.clicked.connect(self._launch_quiz)
        layout.addWidget(btn_quiz)

        layout.addStretch()

    def _launch_facemask(self):
        if self._face_thread and self._face_thread.isRunning():
            QMessageBox.information(self, "Already Running",
                "Face Mask is already open. Close it first (press Esc).")
            return
        self._face_thread = FaceMaskThread()
        self._face_thread.start()

    def _launch_quiz(self):
        if self._quiz_thread and self._quiz_thread.isRunning():
            QMessageBox.information(self, "Already Running",
                "The quiz is already open.")
            return
        self._quiz_thread = QuizThread()
        self._quiz_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())