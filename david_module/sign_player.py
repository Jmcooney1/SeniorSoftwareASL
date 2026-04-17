"""
Reusable widget for embedding sign-language animations anywhere in the app.

Uses Panda3D's ``parent-window`` PRC variable to render directly inside a
native Qt widget — no QPanda3D dependency required.  Falls back to a
subprocess popout on platforms where ``parent-window`` is unsupported.

Usage from any module::

    from david_module.sign_player import SignPlayerWidget

    player = SignPlayerWidget()          # optional: SignPlayerWidget("path/to/sign.csv")
    some_layout.addWidget(player)
    player.play("path/to/sign.csv")      # start or hot-swap to a new sign
    player.play("path/to/other.csv")     # just swaps the clip, no restart
    player.stop()                        # tear down; can play() again later
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QResizeEvent


class SignPlayerWidget(QWidget):
    """Embeds a Panda3D sign animation.  Call ``play(csv_path)`` to start or swap signs."""

    # Panda task-manager step interval in milliseconds (~60 fps)
    _STEP_MS = 16

    def __init__(self, csv_path: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)

        self._base = None             # ShowBase instance (when embedded)
        self._animator = None         # CSVRigAnimator
        self._sign_hud = None
        self._camera_ctrl = None
        self._debug_viz = None
        self._pose_ctrl = None
        self._timer: QTimer | None = None
        self._popout_proc: subprocess.Popen | None = None  # fallback subprocess

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(400)

        # Ensure an OS-native window handle is available for reparenting
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)

        # Placeholder shown before the first play() call
        self._placeholder = QLabel("No sign loaded — select a sign and press Start")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        if self._base is not None:
            # Already running — just swap the clip
            self._swap_clip(csv_path)
            return

        if self._popout_proc is not None:
            # Kill any existing popout before starting a new one
            self._kill_popout()

        # First call — try native embed, fall back to popout
        self._init_panda(csv_path)

    def stop(self) -> None:
        """Tear down the Panda scene so it can be restarted later."""
        self._stop_timer()
        self._kill_popout()

        if self._base is not None:
            try:
                if self._pose_ctrl is not None:
                    self._pose_ctrl.destroy()
                self._base.destroy()
            except Exception:
                pass
            self._base = None
            self._animator = None
            self._sign_hud = None
            self._camera_ctrl = None
            self._debug_viz = None
            self._pose_ctrl = None

        self._placeholder.setText("Stopped — select a sign to play again")
        self._placeholder.setStyleSheet("color: #888; font-style: italic;")
        self._placeholder.setVisible(True)

    @property
    def is_running(self) -> bool:
        """True once the Panda scene has been initialised (embedded or popout)."""
        return self._base is not None or (
            self._popout_proc is not None and self._popout_proc.poll() is None
        )

    # ------------------------------------------------------------------
    # Internal — embedded path (parent-window)
    # ------------------------------------------------------------------

    def _ensure_panda_path(self) -> None:
        pkg_dir = os.path.join(os.path.dirname(__file__), "panda_port")
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

    def _init_panda(self, csv_path: str) -> None:
        """Try to embed Panda3D via ``parent-window``; fall back to popout."""
        self._ensure_panda_path()

        try:
            self._init_embedded(csv_path)
        except Exception as exc:
            # Embedded failed — try popout subprocess as fallback
            self._base = None
            try:
                self._init_popout(csv_path)
                self._placeholder.setText(
                    f"Embedded init failed ({exc}); opened in popout window."
                )
                self._placeholder.setStyleSheet("color: #b45309; font-style: italic;")
                self._placeholder.setVisible(True)
            except Exception as pop_exc:
                self._show_error(
                    f"Embedded: {exc}\nPopout: {pop_exc}"
                )

    def _init_embedded(self, csv_path: str) -> None:
        """Create a ShowBase that renders inside this widget via ``parent-window``."""
        from panda3d.core import loadPrcFileData, WindowProperties
        from direct.showbase.ShowBase import ShowBase

        hwnd = int(self.winId())
        w = max(self.width(), 320)
        h = max(self.height(), 240)

        # Tell ShowBase not to open a window automatically;
        # we'll open one ourselves with the correct parent handle.
        loadPrcFileData("sign-player-embed", "window-type none")

        base = ShowBase()
        base.disableMouse()

        # Now open a window explicitly parented into this QWidget
        props = WindowProperties()
        props.setParentWindow(hwnd)
        props.setOrigin(0, 0)
        props.setSize(w, h)
        props.setUndecorated(True)
        base.openDefaultWindow(props=props)

        base.setBackgroundColor(0.08, 0.09, 0.11, 1)

        # Do NOT let Escape close the host app
        base.accept("escape", lambda: None)

        from panda_core import (
            create_animator,
            create_camera_controller,
            create_character_pose_controller,
            create_sign_hud,
            frame_camera,
            load_actor,
            setup_lighting,
        )
        from landmark_debug import LandmarkVisualizer

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

        animator = create_animator(character, csv_path)
        base.taskMgr.add(animator.update, "csv-rig-animator")

        debug_viz = LandmarkVisualizer(
            base, character,
            camera_controller=camera_ctrl,
            hand_world_space=getattr(animator, "hand_world_space", True),
        )

        def _update_debug(task):
            debug_viz.update(animator.last_pose_lms, animator.last_hand_lms)
            return task.cont

        base.taskMgr.add(_update_debug, "debug-viz-update")

        try:
            pose_ctrl = create_character_pose_controller(base, character, camera=scene_camera)
        except Exception:
            pose_ctrl = None

        sign_hud = create_sign_hud(base, animator)

        # Store everything so we can hot-swap / tear down later
        self._base = base
        self._animator = animator
        self._sign_hud = sign_hud
        self._camera_ctrl = camera_ctrl
        self._debug_viz = debug_viz
        self._pose_ctrl = pose_ctrl

        # Hide the placeholder
        self._placeholder.setVisible(False)

        # Drive Panda's task manager from a QTimer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._panda_step)
        self._timer.start(self._STEP_MS)

    def _panda_step(self) -> None:
        """Advance Panda3D by one frame."""
        if self._base is not None:
            try:
                self._base.taskMgr.step()
            except Exception:
                self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    # ------------------------------------------------------------------
    # Internal — popout fallback (subprocess)
    # ------------------------------------------------------------------

    def _init_popout(self, csv_path: str) -> None:
        """Launch Panda3D as a separate process (works everywhere)."""
        script = os.path.join(os.path.dirname(__file__), "panda_port", "run_panda.py")
        self._popout_proc = subprocess.Popen(
            [sys.executable, script, csv_path],
            cwd=os.path.dirname(script),
        )

    def _kill_popout(self) -> None:
        if self._popout_proc is not None:
            try:
                self._popout_proc.terminate()
            except Exception:
                pass
            self._popout_proc = None

    # ------------------------------------------------------------------
    # Internal — clip swap & resize
    # ------------------------------------------------------------------

    def _swap_clip(self, csv_path: str) -> None:
        """Hot-swap the animation clip without rebuilding the scene."""
        self._ensure_panda_path()

        from animation import CSVSignClip
        from panda_core import update_sign_hud

        clip = CSVSignClip(csv_path)
        self._animator.set_clip(clip)
        update_sign_hud(self._sign_hud, self._animator)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._base is not None and self._base.win is not None:
            from panda3d.core import WindowProperties
            props = WindowProperties()
            props.setOrigin(0, 0)
            props.setSize(event.size().width(), event.size().height())
            self._base.win.requestProperties(props)

    # ------------------------------------------------------------------
    # Internal — error display & cleanup
    # ------------------------------------------------------------------

    def _show_error(self, msg: str) -> None:
        self._placeholder.setText(msg)
        self._placeholder.setStyleSheet("color: #c00; font-style: italic;")
        self._placeholder.setVisible(True)

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)