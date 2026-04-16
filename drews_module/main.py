"""
drews_module/main.py
Exposes get_tab() -> QWidget for the root launcher.
Contains two sub-tabs: ASL Letter Quiz and Face Mask / Hand Tracking.
No tkinter anywhere.
"""
import os
import sys
import random
import io

from PySide6.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

# ── Path resolution ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_paths():
    import json
    search = os.path.abspath(SCRIPT_DIR)
    for _ in range(4):
        candidate = os.path.join(search, "config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                cfg = json.load(f)
            base = cfg.get("dataset_path", "dataSet")
            save = cfg.get("save_dir", "savedVideoPoints")
            if not os.path.isabs(base): base = os.path.join(search, base)
            if not os.path.isabs(save): save = os.path.join(search, save)
            return base, save
        search = os.path.dirname(search)
    root = os.path.dirname(SCRIPT_DIR)
    return os.path.join(root, "dataSet"), os.path.join(root, "savedVideoPoints")


_BASE, SAVE_DIR = _load_paths()
DATA_DIR = os.path.join(_BASE, "drew_dataset", "asl_letters")


# ── Shared button styles ──────────────────────────────────────────────────────
def _btn(bg, fg, hover, press, border="none"):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 13px;
            font-weight: bold;
            border: {border};
        }}
        QPushButton:hover    {{ background: {hover}; color: {fg}; }}
        QPushButton:pressed  {{ background: {press}; color: {fg}; }}
        QPushButton:disabled {{ background: #94a3b8; color: #e2e8f0; border: none; }}
    """

BTN_PRIMARY = _btn("#2563eb", "white", "#1d4ed8", "#1e40af")
BTN_DANGER  = _btn("#dc2626", "white", "#b91c1c", "#991b1b")

# Used for general buttons (e.g. Restart) — has padding
BTN_GHOST = """
    QPushButton {
        background: #f1f5f9;
        color: #1e293b;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: bold;
        border: 1.5px solid #cbd5e1;
    }
    QPushButton:hover    { background: #e2e8f0; color: #0f172a; }
    QPushButton:pressed  { background: #cbd5e1; color: #0f172a; }
    QPushButton:disabled { background: #f8fafc; color: #94a3b8; border-color: #e2e8f0; }
"""

# Used specifically for the 42×42 letter grid — no padding so letter is centered
BTN_LETTER = """
    QPushButton {
        background: #f1f5f9;
        color: #1e293b;
        border-radius: 8px;
        padding: 0px;
        font-size: 14px;
        font-weight: bold;
        border: 1.5px solid #cbd5e1;
    }
    QPushButton:hover    { background: #dbeafe; color: #1d4ed8; border-color: #93c5fd; }
    QPushButton:pressed  { background: #bfdbfe; color: #1e40af; }
    QPushButton:disabled { background: #f8fafc; color: #94a3b8; border-color: #e2e8f0; }
"""

BTN_LETTER_CORRECT = """
    QPushButton {
        background: #dcfce7; color: #166534;
        border-radius: 8px; padding: 0px;
        font-size: 14px; font-weight: bold;
        border: 1.5px solid #86efac;
    }
    QPushButton:disabled { background: #dcfce7; color: #166534; border-color: #86efac; }
"""

BTN_LETTER_WRONG = """
    QPushButton {
        background: #fee2e2; color: #991b1b;
        border-radius: 8px; padding: 0px;
        font-size: 14px; font-weight: bold;
        border: 1.5px solid #fca5a5;
    }
    QPushButton:disabled { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
"""


# ════════════════════════════════════════════════════════════════════════════
#  ASL Letter Quiz tab
# ════════════════════════════════════════════════════════════════════════════
IMG_SIZE = (300, 300)


def _load_questions(folder: str):
    if not os.path.isdir(folder):
        return []
    qs = []
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            continue
        if len(base) != 1 or not base.isalpha():
            continue
        qs.append({"answer": base.upper(), "path": os.path.join(folder, name)})
    random.shuffle(qs)
    return qs


class QuizTab(QWidget):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir  = data_dir
        self.questions = []
        self.remaining = []
        self.current   = None
        self.total     = 0
        self.correct   = 0
        self._build_ui()
        self._load()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 24, 40, 24)
        layout.setSpacing(14)

        # ── Score row + restart ──────────────────────────────────────────
        score_row = QHBoxLayout()
        self.score_label = QLabel("Score: 0 / 0")
        self.score_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1e293b;")
        score_row.addWidget(self.score_label)
        score_row.addStretch()
        restart_btn = QPushButton("↺  Restart")
        restart_btn.setStyleSheet(BTN_GHOST)
        restart_btn.setFixedWidth(110)
        restart_btn.clicked.connect(self._restart)
        score_row.addWidget(restart_btn)
        layout.addLayout(score_row)

        # ── ASL image ───────────────────────────────────────────────────
        self.image_label = QLabel()
        self.image_label.setFixedSize(*IMG_SIZE)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px solid #e2e8f0; border-radius: 12px; background: #f8fafc;")
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── Prompt ──────────────────────────────────────────────────────
        self.prompt_label = QLabel("What letter is this ASL sign?")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setStyleSheet("font-size: 15px; color: #475569;")
        layout.addWidget(self.prompt_label)

        # ── Feedback ────────────────────────────────────────────────────
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #1e293b; min-height: 26px;")
        layout.addWidget(self.feedback_label)

        # ── A–Z letter grid (two rows of 13) ────────────────────────────
        self.letter_buttons: dict[str, QPushButton] = {}
        for row_letters in [
            [chr(i) for i in range(ord('A'), ord('N'))],    # A–M
            [chr(i) for i in range(ord('N'), ord('Z') + 1)], # N–Z
        ]:
            row = QHBoxLayout()
            row.setSpacing(6)
            for letter in row_letters:
                btn = QPushButton(letter)
                btn.setFixedSize(42, 42)
                btn.setStyleSheet(BTN_LETTER)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda checked, l=letter: self._submit(l))
                self.letter_buttons[letter] = btn
                row.addWidget(btn)
            layout.addLayout(row)

        layout.addStretch()

    # ── Data loading ─────────────────────────────────────────────────────────
    def _load(self):
        self.questions = _load_questions(self.data_dir)
        if not self.questions:
            self.prompt_label.setText(
                f"⚠️  No images found in:\n{self.data_dir}\n\nAdd files like A.png, B.png …")
            self._set_buttons_enabled(False)
            return
        self._restart()

    def _restart(self):
        self.remaining = self.questions[:]
        random.shuffle(self.remaining)
        self.total   = 0
        self.correct = 0
        self._update_score()
        self.feedback_label.setText("")
        self.feedback_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #1e293b; min-height: 26px;")
        self._set_buttons_enabled(True)
        self._next()

    def _next(self):
        if not self.remaining:
            self.remaining = self.questions[:]
            random.shuffle(self.remaining)
        self.current = self.remaining.pop()
        self._show_image(self.current["path"])
        self.prompt_label.setText("What letter is this ASL sign?")
        self._reset_button_styles()

    # ── Answer handling ──────────────────────────────────────────────────────
    def _submit(self, letter: str):
        if not self.current:
            return
        correct = self.current["answer"]
        self.total += 1

        if letter == correct:
            self.correct += 1
            self.feedback_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #16a34a;")
            self.feedback_label.setText(f"✅  Correct!  It was {correct}.")
            self.letter_buttons[letter].setStyleSheet(BTN_LETTER_CORRECT)
        else:
            self.feedback_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #dc2626;")
            self.feedback_label.setText(
                f"❌  Wrong — it was {correct}, you picked {letter}.")
            self.letter_buttons[letter].setStyleSheet(BTN_LETTER_WRONG)
            self.letter_buttons[correct].setStyleSheet(BTN_LETTER_CORRECT)

        self._set_buttons_enabled(False)
        self._update_score()
        QTimer.singleShot(900, self._advance)

    def _advance(self):
        self._set_buttons_enabled(True)
        self._next()

    # ── Image display ────────────────────────────────────────────────────────
    def _show_image(self, path: str):
        from PIL import Image
        img = Image.open(path).convert("RGBA").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.read())
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                *IMG_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _update_score(self):
        pct = int(self.correct / self.total * 100) if self.total else 0
        self.score_label.setText(f"Score: {self.correct} / {self.total}  ({pct}%)")

    def _set_buttons_enabled(self, on: bool):
        for b in self.letter_buttons.values():
            b.setEnabled(on)

    def _reset_button_styles(self):
        for b in self.letter_buttons.values():
            b.setEnabled(True)
            b.setStyleSheet(BTN_LETTER)


# ════════════════════════════════════════════════════════════════════════════
#  Face Mask tab
# ════════════════════════════════════════════════════════════════════════════
class FaceMaskThread(QThread):
    error    = Signal(str)
    finished = Signal()

    def run(self):
        try:
            from drews_module.faceMask import run
            run()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class FaceMaskTab(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("😷")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 64px;")
        layout.addWidget(icon)

        title = QLabel("Face Mask & Hand Tracking")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e293b;")
        layout.addWidget(title)

        desc = QLabel(
            "Opens your webcam and overlays a face mask with hand landmark tracking.\n"
            "Press  Esc  inside the camera window to stop."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; color: #64748b;")
        layout.addWidget(desc)

        self.start_btn = QPushButton("▶  Start Camera")
        self.start_btn.setStyleSheet(BTN_PRIMARY)
        self.start_btn.setFixedWidth(200)
        self.start_btn.clicked.connect(self._start)
        layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setStyleSheet(BTN_DANGER)
        self.stop_btn.setFixedWidth(200)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._remind_stop)
        layout.addWidget(self.stop_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: #64748b;")
        layout.addWidget(self.status_label)

    def _start(self):
        if self._thread and self._thread.isRunning():
            return
        self._thread = FaceMaskThread()
        self._thread.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
        self._thread.finished.connect(self._on_done)
        self._thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Camera running — press Esc in the camera window to stop.")

    def _remind_stop(self):
        self.status_label.setText("Press  Esc  inside the camera window to close it.")

    def _on_done(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Camera stopped.")


# ════════════════════════════════════════════════════════════════════════════
#  Drew's module container widget  +  get_tab() entry point
# ════════════════════════════════════════════════════════════════════════════
class DrewsWidget(QWidget):
    """Inner tab widget holding Quiz and Face Mask sub-tabs."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        inner_tabs = QTabWidget()
        inner_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #f8fafc; }
            QTabBar::tab {
                background: #f1f5f9;
                color: #334155;
                padding: 8px 18px;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected  {
                color: #2563eb;
                border-bottom: 2px solid #2563eb;
                background: white;
            }
            QTabBar::tab:hover:!selected { background: #e8eef5; color: #1e293b; }
        """)
        inner_tabs.addTab(QuizTab(DATA_DIR), "🤟  ASL Letter Quiz")
        inner_tabs.addTab(FaceMaskTab(),      "😷  Face Mask")
        layout.addWidget(inner_tabs)


def get_tab() -> QWidget:
    """Called by the root launcher — returns this module's tab content."""
    return DrewsWidget()