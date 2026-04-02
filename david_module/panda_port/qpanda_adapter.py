import math
import os

from direct.actor.Actor import Actor
from panda3d.core import AmbientLight, DirectionalLight, LPoint3f, Vec3, Filename

from camera_controller import FlyCameraController
from landmark_animation import LandmarkRigAnimator

try:
    from QPanda3D.Panda3DWorld import Panda3DWorld
except Exception:  # pragma: no cover - QPanda3D is optional
    Panda3DWorld = None


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain.bam.pz")

ROOT_POS = LPoint3f(0, 0, -1)
ROOT_HPR = Vec3(0, -90, 0)
ROOT_SCALE = 3.0


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

    def __init__(self, width: int = 1024, height: int = 768):
        # Dynamically create a subclass so static type checkers don't fail
        class _Impl(Panda3DWorld):
            pass

        # Initialize the real world instance
        self._world = _Impl(width=width, height=height)

        # Set up the scene similarly to PandaApp
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        model_fname = Filename.fromOsSpecific(MODEL_PATH)
        model_np = self._world.loader.loadModel(model_fname)
        if model_np is None or model_np.isEmpty():
            raise FileNotFoundError(f"Panda failed to load model: {MODEL_PATH}")

        self.character = Actor(model_np)
        self.character.reparentTo(self._world.render)
        self.character.setPos(ROOT_POS)
        self.character.setHpr(ROOT_HPR)
        self.character.setScale(ROOT_SCALE)

        self._setup_lighting()

        try:
            if getattr(self._world, "camLens", None) is not None:
                self._frame_camera()
        except Exception:
            pass

        try:
            self.camera_controller = FlyCameraController(self._world, self._world.cam)
        except Exception:
            self.camera_controller = None
        # Determine which anim subfolder to use. Prefer the setting from
        # panda_main.ANIM_SUBFOLDER if available so embedded and popout match.
        try:
            from panda_main import ANIM_SUBFOLDER as _SUBFOLDER
        except Exception:
            _SUBFOLDER = "act"

        anim_dir = os.path.abspath(os.path.join(BASE_DIR, "anim", _SUBFOLDER))
        if not os.path.isdir(anim_dir):
            raise FileNotFoundError(f"Animation folder not found: {anim_dir}")

        self.landmark_animator = LandmarkRigAnimator(self.character, anim_dir=anim_dir)
        self._world.taskMgr.add(self.landmark_animator.update, "landmark-rig-animator")

    # Proxy attributes/methods commonly used by QPanda3D widget
    def __getattr__(self, item):
        return getattr(self._world, item)

    def _setup_lighting(self) -> None:
        ambient = AmbientLight("ambient-light")
        ambient.setColor((0.52, 0.52, 0.58, 1))
        ambient_np = self._world.render.attachNewNode(ambient)
        self._world.render.setLight(ambient_np)

        key = DirectionalLight("key-light")
        key.setColor((0.75, 0.75, 0.78, 1))
        key_np = self._world.render.attachNewNode(key)
        key_np.setHpr(-20, -18, 0)
        self._world.render.setLight(key_np)

    def _frame_camera(self) -> None:
        bounds = self.character.getTightBounds()
        if not bounds or bounds[0] is None or bounds[1] is None:
            self._world.cam.setPos(0, -12, 1.5)
            self._world.cam.lookAt(0, 0, 1.5)
            return

        min_point, max_point = bounds
        center = (min_point + max_point) * 0.5
        size = max_point - min_point

        try:
            self._world.camLens.setFov(50)
            self._world.camLens.setNearFar(0.1, 1000)

            horizontal_fov, vertical_fov = self._world.camLens.getFov()
            half_width = max(size.x * 0.5, 1.0)
            half_height = max(size.z * 0.5, 1.0)

            distance_for_width = half_width / math.tan(math.radians(horizontal_fov * 0.5))
            distance_for_height = half_height / math.tan(math.radians(vertical_fov * 0.5))
            camera_distance = max(distance_for_width, distance_for_height) * 1.15

            focus_point = LPoint3f(center.x, center.y, center.z + size.z * 0.05)
            camera_point = LPoint3f(center.x, center.y - camera_distance, center.z + size.z * 0.02)

            self._world.cam.setPos(camera_point)
            self._world.cam.lookAt(focus_point)
        except Exception:
            pass
