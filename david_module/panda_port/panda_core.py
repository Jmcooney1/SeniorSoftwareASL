from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from direct.actor.Actor import Actor
from direct.showbase.DirectObject import DirectObject
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Filename,
    LPoint3f,
    Quat,
    TextNode,
    TransformState,
    Vec3,
)


PART_NAME = "modelRoot"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rain.bam.pz"


# Default transform applied to the loaded character.
MODEL_POS = LPoint3f(0, 0, -1)
MODEL_HPR = Vec3(0, -90, 0)

# Simple orbit-camera settings for quick manual tuning.
CAMERA_HEIGHT = 0.42
CAMERA_TARGET_X = 0.0
CAMERA_TARGET_Y = 0.0
CAMERA_TARGET_Z = 0.26
CAMERA_DISTANCE = 1.8
CAMERA_MIN_DISTANCE = 0.85
CAMERA_MAX_DISTANCE = 2.8
CAMERA_INITIAL_AZIMUTH_DEGREES = 0.0
CAMERA_ORBIT_SPEED_DEGREES = 90.0
CAMERA_ZOOM_SPEED = 3.0
CAMERA_FOV_DEGREES = 60.0
CAMERA_NEAR = 0.01
CAMERA_FAR = 100.0

EYE_FOLLOW_TOGGLE_KEY = "f"
EYES_FOLLOW_CAMERA_BY_DEFAULT = True
DEFAULT_EYE_VERTICAL_ANGLE_DEGREES = 0.0
MAX_EYE_TURN_DEGREES = 25.0
EYE_JOINT_NAMES = {
    "L": "DEF-Eye.L",
    "R": "DEF-Eye.R",
}
PONYTAIL_JOINT_HPR_OFFSETS = {
    "FK-Hair_Ponytail1": (0.0, -30.0, 0.0),
    "FK-Hair_Ponytail2": (0.0, -30.0, 0.0),
    "FK-Hair_Ponytail3": (0.0, -30.0, 0.0),
    "FK-Hair_Ponytail4": (0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class AnimationConfig:
    backend: str
    dataset_root: Path | None = None
    gloss: str | None = None
    variant: int | None = None
    clip_path: Path | None = None
    anim_dir: Path | None = None


@dataclass(frozen=True)
class JointRestTransform:
    pos: Vec3
    quat: Quat
    scale: Vec3


def model_filename() -> Filename:
    return Filename.fromOsSpecific(str(MODEL_PATH))


def camera_target_point() -> LPoint3f:
    return LPoint3f(CAMERA_TARGET_X, CAMERA_TARGET_Y, CAMERA_TARGET_Z)


def camera_position(
    distance: float = CAMERA_DISTANCE,
    azimuth_degrees: float = CAMERA_INITIAL_AZIMUTH_DEGREES,
    height: float = CAMERA_HEIGHT,
) -> LPoint3f:
    azimuth_radians = math.radians(azimuth_degrees)
    return LPoint3f(
        CAMERA_TARGET_X + (math.sin(azimuth_radians) * distance),
        CAMERA_TARGET_Y - (math.cos(azimuth_radians) * distance),
        height,
    )


def load_actor(world, pos: LPoint3f = MODEL_POS, hpr: Vec3 = MODEL_HPR) -> Actor:
    """Load the model via the world's loader, parent to the world's render,
    and apply the default transform. Raises FileNotFoundError on load failure.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\n\nPlease place the model file at: {MODEL_PATH}")

    model_fname = model_filename()
    model_np = world.loader.loadModel(model_fname)
    if model_np is None or model_np.isEmpty():
        raise FileNotFoundError(f"Panda failed to load model: {MODEL_PATH}")

    character = Actor(model_np)
    character.reparentTo(world.render)
    character.setPos(pos)
    character.setHpr(hpr)

    return character


def setup_lighting(world) -> None:
    ambient = AmbientLight("ambient-light")
    ambient.setColor((0.52, 0.52, 0.58, 1))
    ambient_np = world.render.attachNewNode(ambient)
    world.render.setLight(ambient_np)

    key = DirectionalLight("key-light")
    key.setColor((0.75, 0.75, 0.78, 1))
    key_np = world.render.attachNewNode(key)
    key_np.setHpr(-20, -18, 0)
    world.render.setLight(key_np)


def frame_camera(world, actor) -> None:
    """Apply the default simple orbit-camera framing."""
    try:
        cam_lens = getattr(world, "camLens", None)
        if cam_lens is None:
            try:
                cam_node = getattr(world, "cam", None)
                if cam_node is not None and getattr(cam_node, "node", None) is not None:
                    cam_lens = cam_node.node().getLens()
            except Exception:
                cam_lens = None

        if cam_lens is None:
            return

        try:
            cam_lens.setFov(CAMERA_FOV_DEGREES)
            cam_lens.setNearFar(CAMERA_NEAR, CAMERA_FAR)
        except Exception:
            return

        camera_node = getattr(world, "camera", None) or getattr(world, "cam", None)
        if camera_node is not None:
            try:
                camera_node.setPos(camera_position())
                camera_node.lookAt(camera_target_point())
            except Exception:
                pass
    except Exception:
        pass


def create_camera_controller(world, camera):
    from camera_controller import CameraController

    return CameraController(
        world,
        camera,
        orbit_center=camera_target_point(),
        distance=CAMERA_DISTANCE,
        camera_height=CAMERA_HEIGHT,
        initial_azimuth_degrees=CAMERA_INITIAL_AZIMUTH_DEGREES,
        orbit_speed_degrees=CAMERA_ORBIT_SPEED_DEGREES,
        zoom_speed=CAMERA_ZOOM_SPEED,
        min_distance=CAMERA_MIN_DISTANCE,
        max_distance=CAMERA_MAX_DISTANCE,
    )


class CharacterPoseController(DirectObject):
    def __init__(
        self,
        base,
        actor: Actor,
        camera=None,
        follow_camera: bool = EYES_FOLLOW_CAMERA_BY_DEFAULT,
    ) -> None:
        super().__init__()
        self.base = base
        self.actor = actor
        self.camera = camera
        self.follow_camera = bool(follow_camera and camera is not None)
        self.task_name = "character-pose-controller"
        self.head_joint = self.actor.exposeJoint(None, PART_NAME, "MSTR-Head_Upper")
        self.eye_world_nodes = {
            side: self.actor.exposeJoint(None, PART_NAME, joint_name)
            for side, joint_name in EYE_JOINT_NAMES.items()
        }
        self.rest_transforms = {
            joint_name: self._joint_rest_transform(joint_name)
            for joint_name in (*PONYTAIL_JOINT_HPR_OFFSETS, *EYE_JOINT_NAMES.values())
        }
        self.eye_rest_angles = {
            side: self._eye_angles_in_head_space(side)
            for side in EYE_JOINT_NAMES
        }

        self._apply_ponytail_pose()
        self._apply_eye_pose()
        self.actor.update()

        self.accept(EYE_FOLLOW_TOGGLE_KEY, self.toggle_follow_camera)
        if self.camera is not None:
            self.base.taskMgr.add(self.update, self.task_name, sort=10)

    def destroy(self) -> None:
        self.ignoreAll()
        if self.camera is not None:
            self.base.taskMgr.remove(self.task_name)

    def toggle_follow_camera(self) -> None:
        if self.camera is None:
            return
        self.follow_camera = not self.follow_camera
        self._apply_eye_pose()
        self.actor.update()

    def update(self, task):
        self._apply_eye_pose()
        self.actor.update()
        return task.cont

    def _joint_rest_transform(self, joint_name: str) -> JointRestTransform:
        local_joint = self.actor.exposeJoint(None, PART_NAME, joint_name, localTransform=1)
        return JointRestTransform(
            pos=Vec3(local_joint.getPos()), # type: ignore
            quat=Quat(local_joint.getQuat()), # type: ignore
            scale=Vec3(local_joint.getScale()), # type: ignore
        )

    def _freeze_joint_with_hpr_offset(self, joint_name: str, hpr_offset: tuple[float, float, float]) -> None:
        rest_transform = self.rest_transforms[joint_name]
        offset_quat = Quat()
        offset_quat.setHpr(hpr_offset)
        self.actor.freezeJoint(
            PART_NAME,
            joint_name,
            transform=TransformState.makePosQuatScale(
                rest_transform.pos,
                rest_transform.quat * offset_quat,
                rest_transform.scale,
            ),
        )

    def _apply_ponytail_pose(self) -> None:
        for joint_name, hpr_offset in PONYTAIL_JOINT_HPR_OFFSETS.items():
            self._freeze_joint_with_hpr_offset(joint_name, hpr_offset)

    def _eye_forward_direction_in_head_space(self, side: str) -> Vec3:
        direction = self.eye_world_nodes[side].getQuat(self.head_joint).xform(Vec3(0, 0, 1)) # type: ignore
        direction.normalize()
        return direction

    def _eye_angles_in_head_space(self, side: str) -> tuple[float, float]:
        direction = self._eye_forward_direction_in_head_space(side)
        horizontal = math.degrees(math.atan2(direction.x, -direction.y))
        vertical = math.degrees(math.atan2(direction.z, -direction.y))
        return horizontal, vertical

    @staticmethod
    def _clamp_angle_offset(angle_offset: float) -> float:
        return max(-MAX_EYE_TURN_DEGREES, min(MAX_EYE_TURN_DEGREES, angle_offset))

    @staticmethod
    def _default_eye_angles() -> tuple[float, float]:
        return 0.0, DEFAULT_EYE_VERTICAL_ANGLE_DEGREES

    def _clamp_target_eye_angles(self, horizontal: float, vertical: float) -> tuple[float, float]:
        default_horizontal, default_vertical = self._default_eye_angles()
        return (
            default_horizontal + self._clamp_angle_offset(horizontal - default_horizontal),
            default_vertical + self._clamp_angle_offset(vertical - default_vertical),
        )

    def _target_eye_angles(self, side: str) -> tuple[float, float]:
        if self.follow_camera and self.camera is not None:
            eye_position = self.eye_world_nodes[side].getPos(self.head_joint) # type: ignore
            camera_position = self.camera.getPos(self.head_joint)
            direction = camera_position - eye_position
            if direction.length_squared() > 0.0:
                direction.normalize()
                horizontal = math.degrees(math.atan2(direction.x, -direction.y))
                vertical = math.degrees(math.atan2(direction.z, -direction.y))
                return self._clamp_target_eye_angles(horizontal, vertical)
        return self._default_eye_angles()

    def _apply_eye_pose(self) -> None:
        for side, joint_name in EYE_JOINT_NAMES.items():
            horizontal_angle, vertical_angle = self._target_eye_angles(side)
            rest_horizontal_angle, rest_vertical_angle = self.eye_rest_angles[side]
            pitch_offset = (
                horizontal_angle - rest_horizontal_angle
                if side == "L"
                else rest_horizontal_angle - horizontal_angle
            )
            roll_offset = (
                vertical_angle - rest_vertical_angle
                if side == "L"
                else rest_vertical_angle - vertical_angle
            )
            self._freeze_joint_with_hpr_offset(
                joint_name,
                (0.0, pitch_offset, roll_offset),
            )


def create_character_pose_controller(world, actor: Actor, camera=None):
    return CharacterPoseController(
        world,
        actor,
        camera=camera,
        follow_camera=EYES_FOLLOW_CAMERA_BY_DEFAULT,
    )



def _sign_label_text(animator) -> str:
    gloss = getattr(animator, "selected_gloss", None)
    clip_path = getattr(animator, "selected_clip_path", None)
    clip_label = clip_path.stem if clip_path is not None else "unknown"
    label = gloss or clip_label
    return f"Signing: {label}"


def _ensure_hud_camera(world) -> None:
    if getattr(world, "_sign_hud_cam2d", None) is not None:
        return
    buffer = getattr(world, "buff", None)
    if buffer is None:
        return
    world._sign_hud_cam2d = world.makeCamera2d(buffer, sort=20)


def create_sign_hud(world, animator):
    parent = getattr(world, "aspect2d", None) or getattr(world, "render2d", None)
    if parent is None:
        return None

    try:
        _ensure_hud_camera(world)
    except Exception:
        pass

    try:
        aspect = float(world.getAspectRatio())
    except Exception:
        aspect = 1.0

    return OnscreenText(
        text=_sign_label_text(animator),
        parent=parent,
        pos=(-aspect + 0.12, 0.92),
        scale=0.055,
        align=TextNode.ALeft,
        fg=(0.96, 0.97, 0.99, 1.0),
        bg=(0.08, 0.1, 0.14, 0.78),
        mayChange=True,
    )


def update_sign_hud(hud, animator) -> None:
    """Refresh the on-screen sign label after a clip swap."""
    if hud is not None:
        hud.setText(_sign_label_text(animator))


def create_animator(actor: Actor, csv_path=None):
    """Create a CSVRigAnimator, optionally pre-loaded with *csv_path*."""
    from animation import CSVRigAnimator
    from pathlib import Path

    return CSVRigAnimator(actor, csv_path=Path(csv_path) if csv_path else None)
