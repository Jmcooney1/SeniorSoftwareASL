import sys
import os
import json
import importlib

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QTabWidget, QStackedWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

# ── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

DATASET_PATH = os.path.join(ROOT_DIR, "dataSet")
SAVE_DIR = os.path.join(ROOT_DIR, "savedVideoPoints")

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"dataset_path": "", "save_dir": ""}

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

def _resolve(path: str) -> str:
    if not os.path.isabs(path):
        return os.path.join(ROOT_DIR, path)
    return path

def config_is_valid(cfg: dict) -> bool:
    base = cfg.get("dataset_path", "")
    return bool(base) and os.path.isdir(_resolve(base))


# ── Module discovery ──────────────────────────────────────────────────────────
def discover_modules() -> list:
    """
    Scans ROOT_DIR for subfolders with __init__.py and main.py.
    module_info.json is optional:  { "name": "...", "emoji": "..." }
    Each module's main.py MUST expose:  def get_tab() -> QWidget
    """
    modules = []
    skip = {"myenv", "build", "dist", "__pycache__", ".git", "dataSet", "savedVideoPoints"}

    for entry in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, entry)
        if not os.path.isdir(folder_path):
            continue
        if entry in skip or entry.startswith("."):
            continue
        if not os.path.exists(os.path.join(folder_path, "__init__.py")):
            continue
        if not os.path.exists(os.path.join(folder_path, "main.py")):
            continue

        info_path = os.path.join(folder_path, "module_info.json")
        try:
            with open(info_path) as f:
                info = json.load(f)
        except Exception:
            info = {}

        modules.append({
            "folder": entry,
            "name":   info.get("name",  entry.replace("_", " ").title()),
            "emoji":  info.get("emoji", "📦"),
        })

    return modules


# ── Styles ────────────────────────────────────────────────────────────────────
BTN_PRIMARY = """
    QPushButton {
        background: #2563eb; color: white;
        border-radius: 8px; padding: 12px 28px;
        font-size: 14px; font-weight: bold;
    }
    QPushButton:hover    { background: #1d4ed8; }
    QPushButton:pressed  { background: #1e40af; }
    QPushButton:disabled { background: #94a3b8; }
"""
BTN_SECONDARY = """
    QPushButton {
        background: #f1f5f9; color: #334155;
        border-radius: 8px; padding: 10px 20px;
        font-size: 13px; border: 1px solid #cbd5e1;
    }
    QPushButton:hover { background: #e2e8f0; }
"""
TAB_STYLE = """
    QTabWidget::pane { border: none; background: #f8fafc; }
    QTabBar::tab {
        background: #e2e8f0; color: #475569;
        padding: 10px 22px; font-size: 13px; font-weight: bold;
        border: none; border-bottom: 3px solid transparent; margin-right: 2px;
    }
    QTabBar::tab:selected  { background: #f8fafc; color: #2563eb; border-bottom: 3px solid #2563eb; }
    QTabBar::tab:hover:!selected { background: #f1f5f9; color: #1e293b; }
"""




# ── Tabbed App Page ───────────────────────────────────────────────────────────
class TabbedAppPage(QWidget):
    """
    The main page. Calls get_tab() on every discovered module and registers
    the returned QWidget as a tab. No child windows are ever opened.
    """
    def __init__(self, on_settings_callback):
        super().__init__()
        self.on_settings = on_settings_callback
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setStyleSheet("background: #1e293b;")
        header.setFixedHeight(54)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        title = QLabel("ASL Translator")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        hl.addWidget(title)
        hl.addStretch()
        settings_btn = QPushButton("⚙️  Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                background: #334155; color: #cbd5e1;
                border-radius: 6px; padding: 6px 14px; font-size: 12px;
                border: 1px solid #475569;
            }
            QPushButton:hover { background: #475569; color: white; }
        """)
        settings_btn.clicked.connect(self.on_settings)
        hl.addWidget(settings_btn)
        outer.addWidget(header)

        # Central tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(TAB_STYLE)
        self._load_tabs()
        outer.addWidget(self.tabs)

    def _load_tabs(self):
        if ROOT_DIR not in sys.path:
            sys.path.insert(0, ROOT_DIR)

        modules = discover_modules()
        if not modules:
            empty = QLabel(
                "No modules found.\n\n"
                "Each module folder needs:\n"
                "  • __init__.py\n"
                "  • main.py  exposing  get_tab() -> QWidget"
            )
            empty.setStyleSheet("color: #64748b; font-size: 14px; padding: 40px;")
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(empty, "No modules")
            return

        for mod in modules:
            label = f"{mod['emoji']}  {mod['name']}"
            try:
                module     = importlib.import_module(f"{mod['folder']}.main")
                tab_widget = module.get_tab()
                self.tabs.addTab(tab_widget, label)
            except AttributeError:
                self.tabs.addTab(self._err(
                    f"{mod['folder']}/main.py needs:\n\n"
                    "def get_tab() -> QWidget:\n    return YourWidget()"
                ), label + " ⚠️")
            except Exception as e:
                self.tabs.addTab(self._err(str(e)), label + " ⚠️")

    @staticmethod
    def _err(msg: str) -> QLabel:
        lbl = QLabel(msg)
        lbl.setStyleSheet("color: #dc2626; font-size: 13px; padding: 30px; font-family: monospace;")
        lbl.setWordWrap(True)
        return lbl


# ── Shell Window ──────────────────────────────────────────────────────────────
class ShellWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Translator")
        self.setMinimumSize(1100, 700)
        self.stack    = QStackedWidget()
        self.app_page = None
        self.setCentralWidget(self.stack)

        # Go straight to the main app (settings page removed for now)
        self._go_home()

    def _go_home(self):
        # Tear down the old page if we're returning from settings
        if self.app_page is not None:
            self.stack.removeWidget(self.app_page)
            self.app_page.deleteLater()

        # ✅ Actually instantiate the page (was commented out before)
        self.app_page = TabbedAppPage(on_settings_callback=self._go_settings)
        self.stack.addWidget(self.app_page)
        self.stack.setCurrentWidget(self.app_page)

    def _go_settings(self):
        # Placeholder — wire up a SettingsPage here when you're ready
        QMessageBox.information(self, "Settings", "Settings page coming soon.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Helvetica Neue", 13))
    window = ShellWindow()
    window.show()
    sys.exit(app.exec())