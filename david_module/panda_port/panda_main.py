import math
import os

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData

from camera_controller import FlyCameraController
from landmark_animation import LandmarkRigAnimator
from panda_shared import load_actor, setup_lighting, frame_camera, get_anim_dir


class PandaApp(ShowBase):
    def __init__(self) -> None:
        loadPrcFileData("", "win-size 1500 900")
        
        super().__init__()
        self.disableMouse()

        self.setBackgroundColor(0.08, 0.09, 0.11, 1)
        self.accept("escape", self.userExit)

        # Load the model and build the scene using shared helpers
        self.character = load_actor(self)
        setup_lighting(self)
        # Only frame camera when a camera/lens exists (may be absent when run with 'window-type none')
        if hasattr(self, 'camLens') and getattr(self, 'camLens', None) is not None:
            try:
                frame_camera(self, self.character)
            except Exception:
                pass
        self.camera_controller = FlyCameraController(self, self.camera)

        # Anim folder lives inside this package (panda_port/anim/<subfolder>)
        anim_dir_to_use = get_anim_dir()
        if not os.path.isdir(anim_dir_to_use):
            raise FileNotFoundError(
                f"Animation folder not found: {anim_dir_to_use}\n\nPlease place 'pose_output.csv' and 'hands_output.csv' inside this folder (set ANIM_SUBFOLDER in panda_shared.py)."
            )

        self.landmark_animator = LandmarkRigAnimator(self.character, anim_dir=anim_dir_to_use)
        self.taskMgr.add(self.landmark_animator.update, "landmark-rig-animator")

    # Lighting and camera framing are provided by panda_shared to avoid duplication
