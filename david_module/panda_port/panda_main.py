import math
import os

from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, LPoint3f, Vec3, loadPrcFileData, Filename

from camera_controller import FlyCameraController
from landmark_animation import LandmarkRigAnimator


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain.bam.pz")

# Name of the subfolder under `anim/` to read CSV files from
ANIM_SUBFOLDER = "act"

ROOT_POS = LPoint3f(0, 0, -1)
ROOT_HPR = Vec3(0, -90, 0)
ROOT_SCALE = 3.0


class PandaApp(ShowBase):
    def __init__(self) -> None:
        loadPrcFileData("", "win-size 1500 900")
        
        super().__init__()
        self.disableMouse()

        self.setBackgroundColor(0.08, 0.09, 0.11, 1)
        self.accept("escape", self.userExit)

        # Ensure the model exists in the local package models/ folder
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n\nPlease place the model file at: {MODEL_PATH}"
            )

        # Load via Panda's Filename API to ensure absolute OS path works correctly
        model_fname = Filename.fromOsSpecific(MODEL_PATH)
        model_np = self.loader.loadModel(model_fname)
        if model_np is None or model_np.isEmpty():
            raise FileNotFoundError(f"Panda failed to load model: {MODEL_PATH}")

        self.character = Actor(model_np)

        self.character.reparentTo(self.render)
        self.character.setPos(ROOT_POS)
        self.character.setHpr(ROOT_HPR)
        self.character.setScale(ROOT_SCALE)

        self._setup_lighting()
        # Only frame camera when a camera/lens exists (may be absent when run with 'window-type none')
        if hasattr(self, 'camLens') and getattr(self, 'camLens', None) is not None:
            try:
                self._frame_camera()
            except Exception:
                pass
        self.camera_controller = FlyCameraController(self, self.camera)

        # Anim folder lives inside this package (panda_port/anim/<subfolder>)
        anim_dir_to_use = os.path.abspath(os.path.join(BASE_DIR, 'anim', ANIM_SUBFOLDER))
        if not os.path.isdir(anim_dir_to_use):
            raise FileNotFoundError(
                f"Animation folder not found: {anim_dir_to_use}\n\nPlease place 'pose_output.csv' and 'hands_output.csv' inside this folder (set ANIM_SUBFOLDER at top of file)."
            )

        self.landmark_animator = LandmarkRigAnimator(self.character, anim_dir=anim_dir_to_use)
        self.taskMgr.add(self.landmark_animator.update, "landmark-rig-animator")

    def _setup_lighting(self) -> None:
        ambient = AmbientLight("ambient-light")
        ambient.setColor((0.52, 0.52, 0.58, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        key = DirectionalLight("key-light")
        key.setColor((0.75, 0.75, 0.78, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(-20, -18, 0)
        self.render.setLight(key_np)

    def _frame_camera(self) -> None:
        bounds = self.character.getTightBounds()
        if not bounds or bounds[0] is None or bounds[1] is None:
            self.camera.setPos(0, -12, 1.5)
            self.camera.lookAt(0, 0, 1.5)
            return

        min_point, max_point = bounds
        center = (min_point + max_point) * 0.5
        size = max_point - min_point

        self.camLens.setFov(50)
        self.camLens.setNearFar(0.1, 1000)

        horizontal_fov, vertical_fov = self.camLens.getFov()
        half_width = max(size.x * 0.5, 1.0)
        half_height = max(size.z * 0.5, 1.0)

        distance_for_width = half_width / math.tan(math.radians(horizontal_fov * 0.5))
        distance_for_height = half_height / math.tan(math.radians(vertical_fov * 0.5))
        camera_distance = max(distance_for_width, distance_for_height) * 1.15

        focus_point = LPoint3f(center.x, center.y, center.z + size.z * 0.05)
        camera_point = LPoint3f(center.x, center.y - camera_distance, center.z + size.z * 0.02)

        self.camera.setPos(camera_point)
        self.camera.lookAt(focus_point)
