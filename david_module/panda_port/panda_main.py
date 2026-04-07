from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData

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


class PandaApp(ShowBase):
    def __init__(self) -> None:
        loadPrcFileData("", "win-size 1200 1000")
        
        super().__init__()
        self.disableMouse()

        self.setBackgroundColor(0.08, 0.09, 0.11, 1)
        self.accept("escape", self.userExit)

        # Load the model and build the scene using shared helpers
        self.character = load_actor(self)
        setup_lighting(self)
        self.scene_camera = getattr(self, "camera", None) or getattr(self, "cam", None)
        # Only frame camera when a camera/lens exists (may be absent when run with 'window-type none')
        if hasattr(self, 'camLens') and getattr(self, 'camLens', None) is not None:
            try:
                frame_camera(self, self.character)
            except Exception:
                pass
        try:
            self.camera_controller = create_camera_controller(self, self.scene_camera) if self.scene_camera else None
        except Exception:
            self.camera_controller = None

        self.landmark_animator = create_animator(self.character)
        self.sign_hud = create_sign_hud(self, self.landmark_animator)
        self.taskMgr.add(self.landmark_animator.update, "landmark-rig-animator")

        # Landmark debug overlay (toggle with V)
        self.debug_viz = LandmarkVisualizer(
            self, self.character,
            camera_controller=self.camera_controller,
            hand_world_space=getattr(self.landmark_animator, "hand_world_space", True),
        )
        self.taskMgr.add(self._update_debug_viz, "debug-viz-update")

        try:
            self.character_pose_controller = create_character_pose_controller(
                self,
                self.character,
                camera=self.scene_camera,
            )
        except Exception:
            self.character_pose_controller = None

    def _update_debug_viz(self, task):
        self.debug_viz.update(
            self.landmark_animator.last_pose_lms,
            self.landmark_animator.last_hand_lms,
        )
        return task.cont
    # Lighting, camera framing, and HUD setup are provided by panda_core to avoid duplication
