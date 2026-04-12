"""
Reusable widget for embedding sign-language animations anywhere in the app.

Usage from any module::

    from david_module.sign_player import SignPlayerWidget

    player = SignPlayerWidget()          # optional: SignPlayerWidget("path/to/sign.csv")
    some_layout.addWidget(player)
    player.play("path/to/sign.csv")      # start or hot-swap to a new sign
    player.play("path/to/other.csv")     # just swaps the clip, no restart
"""

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt


class SignPlayerWidget(QWidget):
    """Embeds a Panda3D sign animation.  Call ``play(csv_path)`` to start or swap signs."""

    def __init__(self, csv_path: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)

        self._world = None            # QPandaPandaWorld adapter
        self._panda_widget = None     # QPanda3DWidget instance

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(400)

        # Placeholder shown before the first play() call
        self._placeholder = QLabel("No sign loaded — call play(csv_path)")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-style: italic;")
        self._layout.addWidget(self._placeholder)

        if csv_path is not None:
            self.play(csv_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, csv_path: str) -> None:
        """Start playing *csv_path*, or hot-swap if already running.

        Parameters
        ----------
        csv_path : str
            Absolute or relative path to a SignSchool CSV file.
        """
        csv_path = str(Path(csv_path).resolve())

        if self._world is not None:
            # Already running — just swap the clip
            self._swap_clip(csv_path)
            return

        # First call — build the full Panda scene
        self._init_panda(csv_path)

    @property
    def is_running(self) -> bool:
        """True once the Panda scene has been initialised."""
        return self._world is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_panda(self, csv_path: str) -> None:
        """Create the QPanda3D world and widget for the first time."""
        pkg_dir = os.path.join(os.path.dirname(__file__), "panda_port")
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

        try:
            from QPanda3D.QPanda3DWidget import QPanda3DWidget
            import qpanda_adapter
        except Exception as exc:
            self._show_error(f"Failed to import QPanda3D: {exc}")
            return

        try:
            self._world = qpanda_adapter.QPandaPandaWorld(
                width=self.width() or 1024,
                height=self.height() or 768,
                csv_path=csv_path,
            )
            real_world = getattr(self._world, "_world", self._world)
            self._panda_widget = QPanda3DWidget(real_world)

            # Remove placeholder and add the Panda widget
            self._placeholder.setVisible(False)
            self._layout.addWidget(self._panda_widget, 1)
        except Exception as exc:
            self._world = None
            self._show_error(f"Embedded start failed: {exc}")

    def _swap_clip(self, csv_path: str) -> None:
        """Hot-swap the animation clip without rebuilding the scene."""
        pkg_dir = os.path.join(os.path.dirname(__file__), "panda_port")
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

        from animation import CSVSignClip
        from panda_core import update_sign_hud

        clip = CSVSignClip(csv_path)
        self._world.landmark_animator.set_clip(clip)
        update_sign_hud(getattr(self._world, "sign_hud", None),
                        self._world.landmark_animator)

    def _show_error(self, msg: str) -> None:
        self._placeholder.setText(msg)
        self._placeholder.setStyleSheet("color: #c00; font-style: italic;")
        self._placeholder.setVisible(True)
