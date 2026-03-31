"""
kily_module/main.py
Exposes get_tab() -> QWidget for the root launcher.
The actual UI lives in kily_module/skeleton_translation.py (unchanged).
"""
import os
import sys

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


_BASE,    SAVE_DIR     = _load_paths()
DB_PATH      = os.path.join(_BASE, "kily_dataset", "wlasl-complete")
VIDEO_FOLDER = os.path.join(DB_PATH, "videos")
VIDEO_INDEX  = os.path.join(DB_PATH, "wlasl_class_list.txt")


def get_tab():
    """Called by the root launcher — returns this module's tab content."""
    from PyQt6.QtWidgets import QWidget
    # skeleton_translation.py defines the full UI as a QWidget subclass.
    # We just import and return it — no QMainWindow, no new window.
    from kily_module.skeleton_translation import TranslatorWidget
    return TranslatorWidget()