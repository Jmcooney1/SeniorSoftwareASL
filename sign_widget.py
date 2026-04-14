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

    def __init__(self, parent_handle: int, width: int, height: int, csv_path: Path | None):
        loadPrcFileData("", "window-type none")
        loadPrcFileData("", "process-events 0")
        loadPrcFileData("", "cocoa-event-loop 0")   # ← stop Panda polling NSApp
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

        self._window_ready = False   # ← guard flag

    def mark_ready(self) -> None:
        self._window_ready = True

    def safe_step(self) -> None:
        """Step the task manager only when the window is fully initialised."""
        if not self._window_ready:
            return
        try:
            self.taskMgr.step()
        except Exception as e:
            print(f"[PandaWorld] step error (ignored): {e}")

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

        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)  # ← reduces AppKit subview nesting
        if sys.platform == "darwin":
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)     # ← stop Qt painting over Panda's layer
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
    # Qt internals
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
        self._shutdown_panda()
        super().closeEvent(event)
        
    def _shutdown_panda(self) -> None:
        # 1. Stop the Qt render timer first
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        # 2. Destroy the Panda3D world cleanly
        if self._world is not None:
            try:
                self._world.taskMgr.stop()        # stop internal task threads
            except Exception:
                pass
            try:
                self._world.destroy()             # shuts down graphics engine, window, etc.
            except Exception:
                pass
            # 3. Remove the global `base` reference ShowBase injects
            try:
                import builtins
                if getattr(builtins, "base", None) is self._world:
                    delattr(builtins, "base")
            except Exception:
                pass
            self._world = None

    def _cleanup_stale_showbase(self) -> None:
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
        self._timer.timeout.connect(self._world.safe_step)   # ← safe_step not taskMgr.step directly
        self._timer.start(16)

        # Give the window one extra frame to settle before we start stepping.
        # This is the window that was getting the bad address — it needs to be
        # fully registered with AppKit before Panda starts calling process_events.
        QTimer.singleShot(150, self._world.mark_ready)

        print("[SignWidget] ✅ render timer started")