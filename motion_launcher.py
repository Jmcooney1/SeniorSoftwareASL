import sys
import os
import json
import importlib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QFrame, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# ── Config ─────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
# We will look for modules specifically in a 'modules' subfolder
MODULES_DIR = os.path.join(ROOT_DIR, "modules") 
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

if not os.path.exists(MODULES_DIR):
    os.makedirs(MODULES_DIR)

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"library_path": "", "output_dir": ""}

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

def config_is_valid(cfg: dict) -> bool:
    # Just check if we have a path set for the motion library
    return bool(cfg.get("library_path"))


# ── Module discovery ────────────────────────────────────────────────────────
def discover_modules() -> list:
    modules = []
    if not os.path.exists(MODULES_DIR):
        return modules

    for entry in sorted(os.listdir(MODULES_DIR)):
        folder_path = os.path.join(MODULES_DIR, entry)

        if not os.path.isdir(folder_path):
            continue

        has_init = os.path.exists(os.path.join(folder_path, "__init__.py"))
        has_main = os.path.exists(os.path.join(folder_path, "main.py"))
        if not (has_init and has_main):
            continue

        info_path = os.path.join(folder_path, "module_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path) as f:
                    info = json.load(f)
            except Exception:
                info = {}
        else:
            info = {}

        modules.append({
            "folder": entry,
            "name":   info.get("name",  entry.replace("_", " ").title()),
            "desc":   info.get("desc",  "Launch motion tool"),
            "emoji":  info.get("emoji", "🚀"),
        })
    return modules


# ── Styles ──────────────────────────────────────────────────────────────────
BTN_PRIMARY = """
    QPushButton {
        background: #2563eb; color: white; border-radius: 8px; 
        padding: 12px 28px; font-size: 14px; font-weight: bold;
    }
    QPushButton:hover { background: #1d4ed8; }
"""
BTN_SECONDARY = """
    QPushButton {
        background: #f1f5f9; color: #334155; border-radius: 8px; 
        padding: 10px 20px; font-size: 13px; border: 1px solid #cbd5e1;
    }
    QPushButton:hover { background: #e2e8f0; }
"""

# ── Settings Page ───────────────────────────────────────────────────────────
class SettingsPage(QWidget):
    def __init__(self, on_save_callback):
        super().__init__()
        self.on_save = on_save_callback
        self._build_ui()
        self._load_existing()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        title = QLabel("⚙️  Motion App Settings")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b;")
        layout.addWidget(title)

        layout.addWidget(self._divider())

        # Field 1: Library Path
        layout.addWidget(self._field_label("📂  Motion Library (.npy)", "Path to your saved ASL motion library file"))
        lib_row = QHBoxLayout()
        self.library_field = QLineEdit()
        self.library_field.setPlaceholderText("Select your .npy library file...")
        self.library_field.setStyleSheet("padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px;")
        browse_lib = QPushButton("Browse...")
        browse_lib.setStyleSheet(BTN_SECONDARY)
        browse_lib.clicked.connect(self._browse_library)
        lib_row.addWidget(self.library_field)
        lib_row.addWidget(browse_lib)
        layout.addLayout(lib_row)

        layout.addStretch()
        
        save_btn = QPushButton("Save & Start Launcher →")
        save_btn.setStyleSheet(BTN_PRIMARY)
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _load_existing(self):
        cfg = load_config()
        self.library_field.setText(cfg.get("library_path", ""))

    def _browse_library(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Motion Library", "", "Numpy Files (*.npy)")
        if file:
            self.library_field.setText(file)

    def _save(self):
        path = self.library_field.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing Info", "Please select a library path.")
            return
        save_config({"library_path": path})
        self.on_save()

    def _field_label(self, title, sub):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        t = QLabel(title)
        t.setStyleSheet("font-size: 14px; font-weight: bold;")
        s = QLabel(sub)
        s.setStyleSheet("font-size: 12px; color: #64748b;")
        v.addWidget(t)
        v.addWidget(s)
        return w

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e2e8f0;")
        return line

# ── Launcher Page ───────────────────────────────────────────────────────────
class LauncherPage(QWidget):
    def __init__(self, on_settings_callback):
        super().__init__()
        self.on_settings = on_settings_callback
        self.child_windows = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QWidget()
        header.setStyleSheet("background: #1e293b;")
        header.setFixedHeight(60)
        hl = QHBoxLayout(header)
        title = QLabel("Motion Launcher")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")
        hl.addWidget(title)
        hl.addStretch()
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setStyleSheet("color: white; background: #334155; padding: 5px 15px; border-radius: 5px;")
        settings_btn.clicked.connect(self.on_settings)
        hl.addWidget(settings_btn)
        outer.addWidget(header)

        # Body
        body = QWidget()
        bl = QVBoxLayout(body)
        modules = discover_modules()

        if not modules:
            bl.addWidget(QLabel("No modules found in /modules folder."))
        else:
            for mod in modules:
                btn = QPushButton(f"{mod['emoji']} {mod['name']}\n{mod['desc']}")
                btn.setStyleSheet("text-align: left; padding: 20px; background: white; border: 1px solid #e2e8f0; border-radius: 10px;")
                btn.clicked.connect(lambda checked, m=mod: self._launch(m))
                bl.addWidget(btn)
        
        bl.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

    def _launch(self, mod: dict):
        # Add the modules folder to sys.path so we can import from it
        if MODULES_DIR not in sys.path:
            sys.path.insert(0, MODULES_DIR)
        
        try:
            # We import the 'main' file from the specific module folder
            module_name = f"{mod['folder']}.main"
            module = importlib.import_module(module_name)
            # Reload to ensure we get fresh code if changed
            importlib.reload(module)
            
            self.win = module.MainWindow()
            self.win.show()
            self.child_windows.append(self.win)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch module: {e}")

# ── Shell Window ────────────────────────────────────────────────────────────
class ShellWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Launcher")
        self.setMinimumSize(700, 600)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self._refresh()

    def _refresh(self):
        self.settings_page = SettingsPage(on_save_callback=self._refresh)
        self.launcher_page = LauncherPage(on_settings_callback=self._go_settings)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.launcher_page)
        
        if config_is_valid(load_config()):
            self.stack.setCurrentWidget(self.launcher_page)
        else:
            self.stack.setCurrentWidget(self.settings_page)

    def _go_settings(self):
        self.stack.setCurrentWidget(self.settings_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShellWindow()
    window.show()
    sys.exit(app.exec())