import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # kily_module/


def _load_paths():
    import json
    search = os.path.abspath(SCRIPT_DIR)
    for _ in range(4):
        candidate = os.path.join(search, "config.json")
        if os.path.exists(candidate):
            project_root = search  # wherever config.json lives = project root
            with open(candidate) as f:
                cfg = json.load(f)
            base = cfg.get("dataset_path", "dataSet")
            save = cfg.get("save_dir", "savedVideoPoints")
            # Resolve relative paths against project root
            if not os.path.isabs(base):
                base = os.path.join(project_root, base)
            if not os.path.isabs(save):
                save = os.path.join(project_root, save)
            return base, save
        search = os.path.dirname(search)
    # Fallback — no config.json found anywhere
    project_root = os.path.dirname(SCRIPT_DIR)
    return (
        os.path.join(project_root, "dataSet"),
        os.path.join(project_root, "savedVideoPoints")
    )


# ── Resolve at import time ──────────────────────────────────────────────────
_BASE,    SAVE_DIR     = _load_paths()
DB_PATH      = os.path.join(_BASE, "kily-dataset", "wlasl-complete")        # dataSet/wlasl-complete
VIDEO_FOLDER = os.path.join(DB_PATH, "videos")
VIDEO_INDEX  = os.path.join(DB_PATH, "wlasl_class_list.txt")

# ── Import the actual window (uses the paths above) ─────────────────────────
from kily_module.skeleton_translation import MainWindow  # noqa: E402