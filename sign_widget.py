from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget

from panda3d.core import WindowProperties, loadPrcFileData
from direct.showbase.ShowBase import ShowBase

from david_module.panda_port.panda_core import (
    load_actor, setup_lighting, frame_camera,
    create_animator, create_camera_controller,
    create_sign_hud, create_character_pose_controller,
    update_sign_hud,
)


class _PandaWorld(ShowBase):
    # ← no setAttribute here, this is a Panda class not a Qt class

    def __init__(self, parent_handle: int, width: int, height: int, csv_path: Path | None):
        loadPrcFileData("", "window-type none")
        loadPrcFileData("", "process-events 0")
        super().__init__(windowType="none")

        props = WindowProperties()
        props.setParentWindow(parent_handle)
        props.setSize(width, height)
        self.openDefaultWindow(props=props)
        print(f"[PandaWorld] win={self.win}  graphicsEngine={self.graphicsEngine}")

        self.disableMouse()
        self.setBackgroundColor(0.08, 0.09, 0.11, 1)

        self.character = load_actor(self)
        setup_lighting(self)

        cam = getattr(self, "camera", None)
        try:
            frame_camera(self, self.character)
        except Exception:
            pass

        try:
            self.camera_ctrl = create_camera_controller(self, cam)
        except Exception:
            self.camera_ctrl = None

        self.animator = create_animator(self.character, csv_path)
        self.taskMgr.add(self.animator.update, "animator")
        self.hud = create_sign_hud(self, self.animator)

        try:
            self.pose_ctrl = create_character_pose_controller(
                self, self.character, camera=cam
            )
        except Exception:
            self.pose_ctrl = None

    def play_sign(self, csv_path: Path) -> None:
        from david_module.panda_port.animation import CSVSignClip
        self.animator.set_clip(CSVSignClip(csv_path))
        update_sign_hud(self.hud, self.animator)


class SignWidget(QWidget):
    """
    Embed an ASL sign animation in any PySide6 layout.

    Usage
    -----
        widget = SignWidget()
        widget = SignWidget("path/to/sign.csv")
        widget.play_sign("path/to/other.csv")
    """

    def __init__(self, csv_path: str | Path | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._csv_path = Path(csv_path) if csv_path else None
        self._world: _PandaWorld | None = None
        self._timer: QTimer | None = None
        self._boot_retries = 0

        # Qt widget attributes — these belong here, on the QWidget
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        if sys.platform == "win32":
            self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)

        self.setMinimumSize(640, 480)

    def play_sign(self, csv_path: str | Path) -> None:
        """Swap to a different sign. Safe to call before or after shown."""
        self._csv_path = Path(csv_path)
        if self._world is not None:
            self._world.play_sign(self._csv_path)

    def stop(self) -> None:
        if self._timer:
            self._timer.stop()

    def resume(self) -> None:
        if self._timer:
            self._timer.start(16)

    # ------------------------------------------------------------------
    # Qt internals — nothing below needs to be called by users
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if self._world is None:
            QTimer.singleShot(200, self._boot_panda)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._world and self._world.win:
            props = WindowProperties()
            props.setSize(self.width(), self.height())
            self._world.win.requestProperties(props)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    def _cleanup_stale_showbase(self) -> None:
        """Destroy any half-initialised ShowBase so we can create a fresh one."""
        try:
            import builtins
            existing = getattr(builtins, "base", None)
            if existing is not None and isinstance(existing, ShowBase):
                print("[SignWidget] destroying stale ShowBase")
                existing.destroy()
                try:
                    delattr(builtins, "base")
                except Exception:
                    pass
        except Exception as e:
            print(f"[SignWidget] cleanup warning: {e}")

    def _boot_panda(self) -> None:
        if self.width() == 0 or self.height() == 0:
            QTimer.singleShot(100, self._boot_panda)
            return

        self._cleanup_stale_showbase()

        try:
            self._world = _PandaWorld(
                parent_handle=int(self.winId()),
                width=self.width(),
                height=self.height(),
                csv_path=self._csv_path,
            )
        except AssertionError:
            self._boot_retries += 1
            if self._boot_retries <= 5:
                delay = 300 * self._boot_retries
                QTimer.singleShot(delay, self._boot_panda)
            else:
                print("[SignWidget] ❌ gave up after 5 retries")
            return
        except Exception as e:
            print(f"[SignWidget] ❌ {e}")
            import traceback; traceback.print_exc()
            return

        self._boot_retries = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._world.taskMgr.step)
        self._timer.start(16)
        print("[SignWidget] ✅ render timer started")