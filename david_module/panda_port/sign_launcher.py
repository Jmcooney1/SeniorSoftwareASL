"""Sign-selection launcher with a dropdown UI shown before the Panda window.

Usage
-----
Run this file directly::

    python sign_launcher.py

A PySide6 dialog lets you choose a **Sign** from the available CSV
(MediaPipe pose_world) data files.

After clicking *Launch*, the Panda3D window opens and plays the selected sign.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------


def _available_csv_signs() -> list[tuple[str, Path]]:
    from animation import list_csv_signs

    return list_csv_signs()


# ---------------------------------------------------------------------------
# Launcher dialog
# ---------------------------------------------------------------------------

class SignLauncherDialog(QDialog):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ASL Sign Launcher")
        self.setFixedSize(380, 160)

        self.result_sign: str | None = None
        self.result_csv_path: Path | None = None

        self._csv_signs: list[tuple[str, Path]] = []
        self._load_sign_lists()
        self._build_ui()

    def _load_sign_lists(self) -> None:
        try:
            self._csv_signs = _available_csv_signs()
        except Exception:
            self._csv_signs = []

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._sign_combo = QComboBox()
        self._sign_combo.setMinimumWidth(240)
        self._sign_combo.addItems([name for name, _ in self._csv_signs])
        form.addRow("Sign:", self._sign_combo)

        layout.addLayout(form)

        self._info_label = QLabel(f"{len(self._csv_signs)} signs available")
        self._info_label.setStyleSheet("color: grey;")
        layout.addWidget(self._info_label)

        buttons = QDialogButtonBox()
        self._launch_btn = buttons.addButton("Launch", QDialogButtonBox.AcceptRole) # pyright: ignore[reportAttributeAccessIssue]
        self._cancel_btn = buttons.addButton("Cancel", QDialogButtonBox.RejectRole) # pyright: ignore[reportAttributeAccessIssue]
        self._launch_btn.clicked.connect(self._on_launch)
        self._cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _on_launch(self) -> None:
        sign = self._sign_combo.currentText().strip()
        if not sign:
            QMessageBox.warning(self, "No sign selected", "Please select a sign from the dropdown.")
            return

        for name, path in self._csv_signs:
            if name == sign:
                self.result_csv_path = path
                break
        else:
            QMessageBox.critical(self, "Not found", f"CSV file for '{sign}' not found.")
            return

        self.result_sign = sign
        self.accept()

    def run(self) -> bool:
        """Show the dialog. Returns True if user clicked Launch."""
        return self.exec() == QDialog.Accepted # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Panda launch
# ---------------------------------------------------------------------------

def _launch_csv(csv_path: Path) -> None:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import loadPrcFileData

    from panda_core import (
        create_camera_controller,
        create_character_pose_controller,
        create_csv_animator,
        create_sign_hud,
        frame_camera,
        load_actor,
        setup_lighting,
    )
    from landmark_debug import LandmarkVisualizer

    print(f"Launching Panda app | pose collection: CSV (MediaPipe) | pose: {csv_path.stem}")
    loadPrcFileData("", "win-size 1200 1000")
    base = ShowBase()
    base.disableMouse()
    base.setBackgroundColor(0.08, 0.09, 0.11, 1)
    base.accept("escape", base.userExit)

    character = load_actor(base)
    setup_lighting(base)

    scene_camera = getattr(base, "camera", None) or getattr(base, "cam", None)
    if hasattr(base, "camLens") and getattr(base, "camLens", None) is not None:
        try:
            frame_camera(base, character)
        except Exception:
            pass
    try:
        camera_ctrl = create_camera_controller(base, scene_camera) if scene_camera else None
    except Exception:
        camera_ctrl = None

    animator = create_csv_animator(character, csv_path)
    hud = create_sign_hud(base, animator)
    base.taskMgr.add(animator.update, "csv-rig-animator")

    # Landmark debug overlay (toggle with V)
    debug_viz = LandmarkVisualizer(
        base, character,
        camera_controller=camera_ctrl,
        hand_world_space=getattr(animator, "hand_world_space", False),
    )

    def _update_debug(task):
        debug_viz.update(animator.last_pose_lms, animator.last_hand_lms)
        return task.cont

    base.taskMgr.add(_update_debug, "debug-viz-update")

    try:
        pose_ctrl = create_character_pose_controller(base, character, camera=scene_camera)
    except Exception:
        pose_ctrl = None

    base.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = SignLauncherDialog()
    if not dialog.run():
        return

    _launch_csv(dialog.result_csv_path) # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    main()
