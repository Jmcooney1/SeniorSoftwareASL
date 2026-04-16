from logging import root
import sys
import os
import importlib
from david_module.sign_player import SignPlayerWidget

from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
    QGraphicsDropShadowEffect, QScrollArea, QSizePolicy, QTabWidget
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QFont, QCursor, QMovie, QPixmap
from PySide6.QtWidgets import QGraphicsOpacityEffect

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SAVE_DIR = os.path.join(ROOT_DIR, "savedVideoPoints")

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Modules ───────────────────────────────────────────────────────────────────
MODULES = {
    "kily_module":  {"name": "Animation",          "emoji": "🎬", "description": "Create animated ASL signs with our interactive tool."},
    "izzy_module":  {"name": "Skeleton Translator", "emoji": "🦴", "description": "Extract skeleton keypoints from video and project motion data."},
    "drews_module": {"name": "Translation Quiz",    "emoji": "📝", "description": "Test your ASL knowledge with an interactive translation quiz."},
}

# ── Categories ────────────────────────────────────────────────────────────────
# To add a new category: add an entry with a label, image filename (in assets/),
# and list of module folder keys from MODULES above.
CATEGORIES = [
    {
        "label":   "Library",
        "image":   "video_full.png",          # assets/video_full.png
        "modules": ["kily_module", "izzy_module"],
    },
    {
        "label":   "Games",
        "image":   "games_banner.png",        # assets/games_banner.png  ← add your image here
        "modules": ["drews_module"],
    },
]
LANDING_IMAGE = "home_banner.png"


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#080c11"
SURFACE   = "#161b22"
BORDER    = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
ACCENT    = "#2f81f7"
ACCENT_HO = "#388bfd"


# ── Styles ────────────────────────────────────────────────────────────────────
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

TAB_ACTIVE = f"""
QPushButton {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 28px;
    font-size: 14px;
    font-weight: 600;
}}
"""

TAB_INACTIVE = f"""
QPushButton {{
    background: transparent;
    color: {TEXT_SEC};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 28px;
    font-size: 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    color: {TEXT_PRI};
    border-color: {ACCENT};
    background: rgba(47,129,247,0.08);
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# SPRING CARD  (width is set dynamically by the grid)
# ──────────────────────────────────────────────────────────────────────────────
class SpringCard(QFrame):
    def __init__(self, mod_key, on_open):
        super().__init__()
        mod = MODULES[mod_key]
        self.mod = {"folder": mod_key, **mod}
        self.on_open = on_open

        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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

        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)

        emoji = QLabel(self.mod["emoji"])
        emoji.setStyleSheet("font-size: 32px; border: none;")
        layout.addWidget(emoji)

        name = QLabel(self.mod["name"])
        name.setStyleSheet(f"color: {TEXT_PRI}; font-weight: bold;")
        layout.addWidget(name)

        desc = QLabel(self.mod["description"])
        desc.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px; border: none;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()

        btn = QPushButton("Open")
        btn.setStyleSheet(BTN_STYLE)
        btn.clicked.connect(lambda: self.on_open(self.mod))
        layout.addWidget(btn)

    def enterEvent(self, event):
        self.shadow.setBlurRadius(30)
        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {ACCENT};
                border-radius: 12px;
            }}
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.shadow.setBlurRadius(0)
        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
        """)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.on_open(self.mod)


# ──────────────────────────────────────────────────────────────────────────────
# HOME SCREEN
# ──────────────────────────────────────────────────────────────────────────────
class HomeScreen(QWidget):
    def __init__(self, on_open):
        super().__init__()
        self.on_open     = on_open
        self.active_cat  = None       # index into CATEGORIES
        self._raw_pixmaps = {}        # cache loaded QPixmaps
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
            # grey placeholder so the layout still works
            pix = QPixmap(800, 400)
            pix.fill(Qt.darkGray)
        self._raw_pixmaps[filename] = pix
        return pix
    
    
    # ── build static skeleton ─────────────────────────────────────────────────
    def _build(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── IMAGE AREA (stretches with window) ──────────────────────────────
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet(f"background: {BG};")
        self.logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(self.logo_label, stretch=6)

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
        self.mod_instance = None

        layout = QVBoxLayout(self)

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
        if current is self.home:
            return

        # ── REFRESH/CLEANUP ──
        try:
            target = getattr(current, "mod_instance", None)
            if target:
                # This correctly cleans up both sub-tabs in main.py
                tabs = target.findChild(QTabWidget)
                if tabs:
                    for i in range(tabs.count()):
                        w = tabs.widget(i)
                        if hasattr(w, '_stop_cam'): w._stop_cam()
                        if hasattr(w, '_stop_camera'): w._stop_camera()
                
                # Direct check if not using tabs
                if hasattr(target, '_stop_cam'): target._stop_cam()
                if hasattr(target, '_stop_camera'): target._stop_camera()

            # Panda3D Cleanup
            try:
                from david_module.panda_port.sign_widget import SignWidget
                for panda_widget in current.findChildren(SignWidget):
                    panda_widget.stop()
                    panda_widget._shutdown_panda()
            except:
                pass
        except Exception as e:
            print(f"Cleanup warning: {e}")

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
        self.navigator = AppNavigator()
        self.setCentralWidget(self.navigator)

    def closeEvent(self, event):
        self.navigator.go_home()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Helvetica Neue", 13))

    w = ShellWindow()
    w.show()

    sys.exit(app.exec())