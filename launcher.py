import sys
import os
import json
import importlib

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# ── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

def load_config() -> dict:
    """Reads the local config file safely."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Now defaults with both required paths
    return {"dataset_path": "", "library_path": ""}

def save_config(cfg: dict):
    """Saves settings to config.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

def _resolve(path: str) -> str:
    """Handles both relative and absolute paths."""
    if not path:
        return ""
    if not os.path.isabs(path):
        return os.path.join(ROOT_DIR, path)
    return path

def config_is_valid(cfg: dict) -> bool:
    """Checks if both the dataset directory and library file exist."""
    ds_ok = bool(cfg.get("dataset_path")) and os.path.isdir(_resolve(cfg["dataset_path"]))
    lib_ok = bool(cfg.get("library_path")) and os.path.isfile(_resolve(cfg["library_path"]))
    return ds_ok and lib_ok


# ── Module discovery ──────────────────────────────────────────────────────────
def discover_modules() -> list:
    modules = []
    skip = {".env", "env", "venv", ".git", "__pycache__", "build", "dist", "download_imges"}
    search_dirs = [ROOT_DIR]
    izzy_path = os.path.join(ROOT_DIR, "izzy_module")
    if os.path.isdir(izzy_path):
        search_dirs.append(izzy_path)

    for s_dir in search_dirs:
        if not os.path.exists(s_dir): continue
        for entry in sorted(os.listdir(s_dir)):
            folder_path = os.path.join(s_dir, entry)
            if not os.path.isdir(folder_path) or entry in skip or entry.startswith("."):
                continue
            if (os.path.exists(os.path.join(folder_path, "__init__.py")) and 
                os.path.exists(os.path.join(folder_path, "main.py"))):
                
                info_path = os.path.join(folder_path, "module_info.json")
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                except:
                    info = {}

                import_prefix = "izzy_module." if "izzy_module" in s_dir else ""
                modules.append({
                    "import_path": f"{import_prefix}{entry}.main",
                    "name": info.get("name", entry.replace("_", " ").title()),
                    "emoji": info.get("emoji", "📦")
                })
    return modules

# ── Styles ────────────────────────────────────────────────────────────────────
BTN_PRIMARY = "background: #2563eb; color: white; border-radius: 8px; padding: 12px; font-weight: bold;"
BTN_SECONDARY = "background: #f1f5f9; color: #334155; border-radius: 6px; padding: 8px; border: 1px solid #cbd5e1;"

# ── Pages ─────────────────────────────────────────────────────────────────────
class SettingsPage(QWidget):
    def __init__(self, on_save):
        super().__init__()
        self.on_save = on_save
        self.init_ui()
        self.load_existing()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(15)

        title = QLabel("⚙️ Project Configuration")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Dataset Field
        layout.addWidget(QLabel("📁 Dataset Folder Path:"))
        ds_layout = QHBoxLayout()
        self.ds_input = QLineEdit()
        self.ds_input.setPlaceholderText("Select folder containing your gesture categories...")
        ds_browse = QPushButton("Browse Folder")
        ds_browse.setStyleSheet(BTN_SECONDARY)
        ds_browse.clicked.connect(self.browse_dataset)
        ds_layout.addWidget(self.ds_input)
        ds_layout.addWidget(ds_browse)
        layout.addLayout(ds_layout)

        # Library Field
        layout.addWidget(QLabel("📄 Gesture Library File (.npy):"))
        lib_layout = QHBoxLayout()
        self.lib_input = QLineEdit()
        self.lib_input.setPlaceholderText("Select your asl_motion_library.npy file...")
        lib_browse = QPushButton("Browse File")
        lib_browse.setStyleSheet(BTN_SECONDARY)
        lib_browse.clicked.connect(self.browse_library)
        lib_layout.addWidget(self.lib_input)
        lib_layout.addWidget(lib_browse)
        layout.addLayout(lib_layout)

        layout.addSpacing(20)
        
        save_btn = QPushButton("Save & Launch App")
        save_btn.setStyleSheet(BTN_PRIMARY)
        save_btn.clicked.connect(self.handle_save)
        layout.addWidget(save_btn)
        layout.addStretch()

    def browse_dataset(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", ROOT_DIR)
        if dir_path:
            self.ds_input.setText(dir_path)

    def browse_library(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Library File", ROOT_DIR, "Numpy Files (*.npy)")
        if file_path:
            self.lib_input.setText(file_path)

    def load_existing(self):
        cfg = load_config()
        self.ds_input.setText(cfg.get("dataset_path", ""))
        self.lib_input.setText(cfg.get("library_path", ""))

    def handle_save(self):
        ds = self.ds_input.text().strip()
        lib = self.lib_input.text().strip()
        
        if not os.path.isdir(_resolve(ds)):
            QMessageBox.critical(self, "Invalid Directory", f"Dataset path not found:\n{ds}")
            return
        if not os.path.isfile(_resolve(lib)):
            QMessageBox.critical(self, "Invalid File", f"Library (.npy) file not found:\n{lib}")
            return

        save_config({"dataset_path": ds, "library_path": lib})
        self.on_save()

class TabbedAppPage(QWidget):
    def __init__(self, on_settings):
        super().__init__()
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { background: #e2e8f0; padding: 10px 20px; font-weight: bold; } QTabBar::tab:selected { background: #f8fafc; color: #2563eb; }")
        
        for mod in discover_modules():
            try:
                m = importlib.import_module(mod['import_path'])
                self.tabs.addTab(m.get_tab(), f"{mod['emoji']} {mod['name']}")
            except Exception as e:
                self.tabs.addTab(QLabel(f"Error loading {mod['name']}: {e}"), f"{mod['name']} ⚠️")
        
        sett_btn = QPushButton("⚙️ Update Settings")
        sett_btn.clicked.connect(on_settings)
        layout.addWidget(self.tabs)
        layout.addWidget(sett_btn)

class ShellWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Translator - Unified Launcher")
        self.setMinimumSize(1000, 700)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.refresh_ui()

    def refresh_ui(self):
        # Clear stack
        while self.stack.count():
            widget = self.stack.takeAt(0).widget()
            if widget: widget.deleteLater()

        if config_is_valid(load_config()):
            self.stack.addWidget(TabbedAppPage(self.show_settings))
        else:
            self.stack.addWidget(SettingsPage(self.refresh_ui))

    def show_settings(self):
        self.stack.addWidget(SettingsPage(self.refresh_ui))
        self.stack.setCurrentIndex(self.stack.count() - 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Cleaner look
    window = ShellWindow()
    window.show()
    sys.exit(app.exec())