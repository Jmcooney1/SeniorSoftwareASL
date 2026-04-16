from logging import root
import sys
import os
import importlib
from david_module.sign_player import SignPlayerWidget

from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
    QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QFont, QCursor, QMovie, QPixmap


# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SAVE_DIR = os.path.join(ROOT_DIR, "savedVideoPoints")

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Modules ───────────────────────────────────────────────────────────────────
MODULES = [
    {
        "folder": "kily_module",
        "name": "Animation",
        "emoji": "🎬",
        "description": "Create animated ASL signs with our interactive tool.",
    },
    {
        "folder": "izzy_module",
        "name": "Skeleton Translator",
        "emoji": "🦴",
        "description": "Extract skeleton keypoints from video and project motion data.",
    },
    {
        "folder": "drews_module",
        "name": "Translation Quiz",
        "emoji": "📝",
        "description": "Test your ASL knowledge with an interactive translation quiz.",
    },
]


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#080c11"
SURFACE   = "#161b22"
BORDER    = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
ACCENT    = "#2f81f7"
ACCENT_HO = "#388bfd"


# ── BUTTON (TRANSPARENT OUTLINE STYLE) ────────────────────────────────────────
BTN_STYLE = f"""
QPushButton {{
    background: transparent;
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    border: 1px solid {ACCENT};
    color: white;
    background: rgba(47, 129, 247, 0.08);
}}
QPushButton:pressed {{
    background: rgba(47, 129, 247, 0.18);
}}
"""

BACK_BTN_STYLE = f"""
QPushButton {{
    background: transparent;
    color: {TEXT_SEC};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
}}
QPushButton:hover {{
    color: {TEXT_PRI};
    border-color: {ACCENT};
    background: rgba(47,129,247,0.08);
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# SPRING CARD
# ──────────────────────────────────────────────────────────────────────────────
class SpringCard(QFrame):
    def __init__(self, mod, on_open):
        super().__init__()

        self.mod = mod
        self.on_open = on_open

        self.setFixedSize(300, 200)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
        """)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(0)
        self.setGraphicsEffect(self.shadow)

        self.anim = QPropertyAnimation(self, b"size")
        self.anim.setDuration(420)
        self.anim.setEasingCurve(QEasingCurve.OutElastic)

        self.base = QSize(300, 200)
        self.hover = QSize(320, 215)

        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)

        emoji = QLabel(self.mod["emoji"])
        emoji.setStyleSheet("font-size: 32px;")
        layout.addWidget(emoji)

        name = QLabel(self.mod["name"])
        name.setStyleSheet(f"color: {TEXT_PRI}; font-weight: bold;")
        layout.addWidget(name)

        desc = QLabel(self.mod["description"])
        desc.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()

        btn = QPushButton("Open")
        btn.setStyleSheet(BTN_STYLE)
        btn.clicked.connect(lambda: self.on_open(self.mod))
        layout.addWidget(btn)

    def enterEvent(self, event):
        self._animate(self.hover)
        self.shadow.setBlurRadius(30)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate(self.base)
        self.shadow.setBlurRadius(0)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.on_open(self.mod)

    def _animate(self, target):
        self.anim.stop()
        self.anim.setStartValue(self.size())
        self.anim.setEndValue(target)
        self.anim.start()


# ──────────────────────────────────────────────────────────────────────────────
# HOME SCREEN (FULL SCREEN + 60/40 SPLIT)
# ──────────────────────────────────────────────────────────────────────────────
class HomeScreen(QWidget):
    def __init__(self, on_open):
        super().__init__()
        self.on_open = on_open
        self.setStyleSheet(f"background: {BG};")
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── TOP 60% (LOGO LAYER) ─────────────────────────────
        top = QWidget()
        top.setStyleSheet(f"background: {BG};")

        top_layout = QVBoxLayout(top)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_path = os.path.join(APP_DIR, "assets", "video_full.png")  # FIX: removed duplicate line

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if os.path.exists(img_path):
            pix = QPixmap(img_path)

            pix = pix.scaled(600, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pix)  # FIX: was missing — pixmap was scaled but never applied to the label
        else:
            fallback = QPixmap(os.path.join(APP_DIR,"assets","logo.png"))
            fallback = fallback.scaled(700, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(fallback)

        top_layout.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

        # ── BOTTOM 40% ─────────────────────────────
        bottom = QWidget()
        bottom.setStyleSheet(f"background: {BG};")

        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(40, 10, 40, 40)

        title = QLabel("Choose a tool")
        title.setStyleSheet(f"font-size: 24px; color: {TEXT_PRI};")
        bottom_layout.addWidget(title)

        sub = QLabel("Select a module below to get started.")
        sub.setStyleSheet(f"color: {TEXT_SEC};")
        bottom_layout.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(20)

        for i, mod in enumerate(MODULES):
            grid.addWidget(SpringCard(mod, self.on_open), i // 3, i % 3)

        bottom_layout.addLayout(grid)

        # add full screen usage
        root.addWidget(top, 6)     # 60%
        root.addWidget(bottom, 4)  # 40%


# ──────────────────────────────────────────────────────────────────────────────
# MODULE VIEW
# ──────────────────────────────────────────────────────────────────────────────
class ModuleView(QWidget):
    def __init__(self, mod, on_back):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")

        layout = QVBoxLayout(self)

        # ── header ──
        header = QWidget()
        header.setFixedHeight(54)
        header.setStyleSheet(f"background: {SURFACE}; border-bottom: 1px solid {BORDER};")

        hl = QHBoxLayout(header)

        back = QPushButton("← Home")
        back.setStyleSheet(BACK_BTN_STYLE)
        back.clicked.connect(on_back)

        label = QLabel(f"{mod['emoji']} {mod['name']}")
        label.setStyleSheet(f"color: {TEXT_PRI}; font-weight: bold;")

        hl.addWidget(back)
        hl.addWidget(label)
        hl.addStretch()

        layout.addWidget(header)

        # ── LOAD REAL MODULE UI ──
        layout.addWidget(self._load_module(mod))

    def _load_module(self, mod: dict) -> QWidget:
        try:
            if APP_DIR not in sys.path:
                sys.path.insert(0, APP_DIR)

            module = importlib.import_module(f"{mod['folder']}.main")
            return module.get_tab()

        except Exception as e:
            err = QLabel(f"Failed to load module:\n{e}")
            err.setStyleSheet("color: red; padding: 40px;")
            return err


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATOR
# ──────────────────────────────────────────────────────────────────────────────
class AppNavigator(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.home = HomeScreen(self.open_module)
        self.addWidget(self.home)

    def open_module(self, mod):
        view = ModuleView(mod, self.go_home)
        self.addWidget(view)
        self.setCurrentWidget(view)

    def go_home(self):
        current = self.currentWidget()
        self.setCurrentWidget(self.home)
        if current is not self.home:
            for sign_widget in current.findChildren(SignPlayerWidget):
                sign_widget.stop()
                sign_widget.setParent(None)
            self.removeWidget(current)
            current.deleteLater()  


# ──────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────────────
class ShellWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Translator")
        self.setMinimumSize(1100, 700)
        self.setCentralWidget(AppNavigator())

    def closeEvent(self, event):
        for widget in self.findChildren(SignPlayerWidget):
            widget.closeEvent(self.event)
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Helvetica Neue", 13))

    w = ShellWindow()
    w.show()

    sys.exit(app.exec())