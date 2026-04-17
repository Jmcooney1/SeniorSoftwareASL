from logging import root
import sys
import os
import importlib
from david_module.sign_player import SignPlayerWidget

from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
    QGraphicsDropShadowEffect, QScrollArea, QSizePolicy
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
    "kily_module":  {"name": "Sign Player",         "emoji": "🤟", "description": "Watch a 3D character perform ASL signs."},
    "izzy_module":  {"name": "Skeleton Translator", "emoji": "🦴", "description": "Extract skeleton keypoints from video and project motion data."},
    "drews_module": {"name": "Translation Quiz",    "emoji": "📝", "description": "Test your ASL knowledge with an interactive translation quiz."},
    "jace_module_quiz": {"name": "Sign Quiz",   "emoji": "❓", "description": "Watch a sign and type what you think it means."},
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
        "modules": ["drews_module", "jace_module_quiz"],
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
        emoji.setStyleSheet("font-size: 32px;")
        layout.addWidget(emoji)

        name = QLabel(self.mod["name"])
        name.setStyleSheet(f"color: {TEXT_PRI}; font-weight: bold; font-size: 15px;")
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

    # ── preload pixmaps ───────────────────────────────────────────────────────
    def _load_pixmap(self, filename):
        if filename in self._raw_pixmaps:
            return self._raw_pixmaps[filename]
        path = os.path.join(APP_DIR, "assets", filename)
        if os.path.exists(path):
            pix = QPixmap(path)
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

        # ── BOTTOM PANEL ─────────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setStyleSheet(f"background: {BG};")
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(40, 14, 40, 30)
        bottom_layout.setSpacing(14)
        
        tab_row = QHBoxLayout()
        tab_row.setSpacing(10)

        self.tab_buttons = []

        # LEFT STRETCH (important)
        tab_row.addStretch()

        for i, cat in enumerate(CATEGORIES):
            btn = QPushButton(cat["label"])
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.clicked.connect(lambda checked, idx=i: self._switch_category(idx))
            tab_row.addWidget(btn)
            self.tab_buttons.append(btn)

        # RIGHT STRETCH (important)
        tab_row.addStretch()
        bottom_layout.addLayout(tab_row)

        # cards container (we rebuild its children on category switch)
        self.cards_container = QFrame()
        self.cards_container.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 16px;
            }}
        """)
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setSpacing(20)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.cards_container)
        self.cards_container.setVisible(False) 
        
        # ── animation setup ─────────────────────────────
        self.cards_anim = QPropertyAnimation(self.cards_container, b"pos")
        self.cards_anim.setDuration(350)
        self.cards_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.cards_opacity = QGraphicsOpacityEffect(self.cards_container)
        self.cards_container.setGraphicsEffect(self.cards_opacity)

        self.opacity_anim = QPropertyAnimation(self.cards_opacity, b"opacity")
        self.opacity_anim.setDuration(350)
        self.opacity_anim.setStartValue(0)
        self.opacity_anim.setEndValue(1)
        self.opacity_anim.setEasingCurve(QEasingCurve.OutCubic)

        root_layout.addWidget(bottom, stretch=4)

        # initial render
        self._current_pix = self._load_pixmap(LANDING_IMAGE)
        for btn in self.tab_buttons:
            btn.setStyleSheet(TAB_INACTIVE)

    # ── switch category ───────────────────────────────────────────────────────
    def _switch_category(self, idx):
        # If clicking the already-active tab, collapse back to landing state
        if self.active_cat == idx:
            self.active_cat = None
            for btn in self.tab_buttons:
                btn.setStyleSheet(TAB_INACTIVE)
            self.cards_container.setVisible(False)
            # restore landing image
            self._current_pix = self._load_pixmap(LANDING_IMAGE)
            self._refresh_image()
            return

        self.active_cat = idx

        # update tab styles
        for i, btn in enumerate(self.tab_buttons):
            btn.setStyleSheet(TAB_ACTIVE if i == idx else TAB_INACTIVE)

        # swap to this category's image
        cat = CATEGORIES[idx]
        self._current_pix = self._load_pixmap(cat["image"])
        self._refresh_image()

        # rebuild cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for mod_key in cat["modules"]:
            if mod_key in MODULES:
                card = SpringCard(mod_key, self.on_open)
                self.cards_layout.addWidget(card)

        self.cards_layout.addStretch()
        self.cards_container.setVisible(True)  # reveal cards

    # ── scale image to current label size ────────────────────────────────────
    def _refresh_image(self):
        if not hasattr(self, "_current_pix"):
            return
        w = self.logo_label.width()
        h = self.logo_label.height()
        if w < 10 or h < 10:
            return
        scaled = self._current_pix.scaled(
            w, h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.logo_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_image()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_image()


# ──────────────────────────────────────────────────────────────────────────────
# MODULE VIEW
# ──────────────────────────────────────────────────────────────────────────────
class ModuleView(QWidget):
    def __init__(self, mod, on_back):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")

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