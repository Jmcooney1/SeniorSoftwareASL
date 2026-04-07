from panda_core import (
    create_animator,
    create_camera_controller,
    create_character_pose_controller,
    create_sign_hud,
    frame_camera,
    load_actor,
    setup_lighting,
)

try:
    from QPanda3D.Panda3DWorld import Panda3DWorld
except Exception:  # pragma: no cover - QPanda3D is optional
    Panda3DWorld = None


class QPandaPandaWorld:
    """Adapter wrapper for embedding the existing Panda scene into QPanda3D.

    This module exposes `QPandaPandaWorld` only when QPanda3D is installed; it
    intentionally avoids importing QPanda3D at module import time for the
    popout use-case.
    """

    def __new__(cls, *args, **kwargs):
        if Panda3DWorld is None:
            raise RuntimeError("QPanda3D is not installed")
        return super().__new__(cls)

    def __init__(self, width: int = 1920, height: int = 1080):
        # Dynamically create a subclass so static type checkers don't fail
        class _Impl(Panda3DWorld):
            pass

        # Initialize the real world instance
        self._world = _Impl(width=width, height=height)

        # Build the scene using the shared helpers so settings live in one place
        self.character = load_actor(self._world)
        setup_lighting(self._world)
        self.scene_camera = getattr(self._world, "camera", None) or getattr(self._world, "cam", None)

        try:
            if getattr(self._world, "camLens", None) is not None:
                frame_camera(self._world, self.character)
        except Exception:
            pass

        try:
            self.camera_controller = create_camera_controller(self._world, self.scene_camera) if self.scene_camera else None
        except Exception:
            self.camera_controller = None

        self.landmark_animator = create_animator(self.character)
        self.sign_hud = create_sign_hud(self._world, self.landmark_animator)
        self._world.taskMgr.add(self.landmark_animator.update, "landmark-rig-animator")
        try:
            self.character_pose_controller = create_character_pose_controller(
                self._world,
                self.character,
                camera=self.scene_camera,
            )
        except Exception:
            self.character_pose_controller = None

    # Proxy attributes/methods commonly used by QPanda3D widget
    def __getattr__(self, item):
        return getattr(self._world, item)
    # Lighting and camera framing are provided by panda_core to avoid duplication
