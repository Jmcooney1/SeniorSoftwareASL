from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from direct.actor.Actor import Actor
from panda3d.core import LMatrix3f, Quat, TransformState, Vec3

if TYPE_CHECKING:
    from panda_core import AnimationConfig


PART_NAME = "modelRoot"
DEFAULT_FPS = 30.0
EPSILON = 1.0e-6
MIN_POSE_CONFIDENCE = 0.2
MIN_HAND_CONFIDENCE = 0.5
TORSO_DEPTH_SIGN = -1.0
HAND_CURL_SIGN = -1.0
TARGET_BASIS_BLEND = 0.3
FINGER_POSE_STRENGTH = 0.45
THUMB_POSE_STRENGTH = 0.35
ARM_DIRECTION_BLEND = 0.35
FINGER_BASE_SWAY_SCALE = 0.2
FINGER_TIP_SWAY_SCALE = 0.0
MAX_FINGER_CURL_RADIANS = math.radians(95.0)

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
LEFT_HAND_OFFSET = POSE_LANDMARK_COUNT
RIGHT_HAND_OFFSET = POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT
TOTAL_LANDMARK_COUNT = POSE_LANDMARK_COUNT + (2 * HAND_LANDMARK_COUNT)

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

SIDES = ("L", "R")

ARM_CONTROLLER_JOINTS = {
    "Upperarm": ("FK-Upperarm", "FK-Forearm"),
    "Forearm": ("FK-Forearm", "FK-Hand"),
}

FINGER_LANDMARKS = {
    "Thumb": ((1, 2), (2, 3), (3, 4)),
    "Index": ((5, 6), (6, 7), (7, 8)),
    "Middle": ((9, 10), (10, 11), (11, 12)),
    "Ring": ((13, 14), (14, 15), (15, 16)),
    "Pinky": ((17, 18), (18, 19), (19, 20)),
}

FINGER_CONTROLLER_JOINTS = {
    "Thumb": ("FK-Thumb1", "FK-Thumb2", "FK-Thumb3"),
    "Index": ("FK-Index1", "FK-Index2", "FK-Index3"),
    "Middle": ("FK-Middle1", "FK-Middle2", "FK-Middle3"),
    "Ring": ("FK-Ring1", "FK-Ring2", "FK-Ring3"),
    "Pinky": ("FK-Pinky1", "FK-Pinky2", "FK-Pinky3"),
}

FINGER_CHILD_CONTROLLER_JOINTS = {
    "Thumb": ("FK-Thumb2", "FK-Thumb3", None),
    "Index": ("FK-Index2", "FK-Index3", None),
    "Middle": ("FK-Middle2", "FK-Middle3", None),
    "Ring": ("FK-Ring2", "FK-Ring3", None),
    "Pinky": ("FK-Pinky2", "FK-Pinky3", None),
}


@dataclass(frozen=True)
class JointRestTransform:
    pos: Vec3
    quat: Quat
    scale: Vec3


@dataclass(frozen=True)
class ArmJointControl:
    joint_name: str
    rest_transform: JointRestTransform
    rest_segment_direction_local: Vec3


@dataclass(frozen=True)
class FingerJointControl:
    joint_name: str
    rest_transform: JointRestTransform
    rest_segment_direction_local: Vec3
    curl_axis_local: Vec3 | None = None
    curl_sign: float = 1.0


def _copy_vec(vec: Vec3) -> Vec3:
    return Vec3(vec.x, vec.y, vec.z)


def _copy_quat(quat: Quat) -> Quat:
    return Quat(quat)


def _mediapipe_to_rig_space(x: float, y: float, z: float) -> Vec3:
    return Vec3(x, -y, z)


def _normalized(vec: Vec3 | None) -> Vec3 | None:
    if vec is None:
        return None
    candidate = _copy_vec(vec)
    if candidate.length_squared() <= EPSILON:
        return None
    candidate.normalize()
    return candidate


def _average_vectors(*vectors: Vec3 | None) -> Vec3 | None:
    valid_vectors = [vector for vector in vectors if vector is not None]
    if not valid_vectors:
        return None
    total = Vec3(0, 0, 0)
    for vector in valid_vectors:
        total += vector
    total *= 1.0 / len(valid_vectors)
    return total


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _lerp_vec(start: Vec3, end: Vec3, amount: float) -> Vec3:
    return (start * (1.0 - amount)) + (end * amount)


def _blend_direction(rest_dir: Vec3, target_dir: Vec3, amount: float) -> Vec3 | None:
    rest = _normalized(rest_dir)
    target = _normalized(target_dir)
    if rest is None:
        return target
    if target is None:
        return rest
    if amount >= 0.999:
        return target
    if amount <= 0.001:
        return rest
    blended = (rest * (1.0 - amount)) + (target * amount)
    return _normalized(blended) or target


def _rotation_from_to(rest_dir: Vec3, target_dir: Vec3) -> Quat:
    rest = _normalized(rest_dir)
    target = _normalized(target_dir)
    if rest is None or target is None:
        return Quat.identQuat()

    dot_product = _clamp(rest.dot(target), -1.0, 1.0)
    if dot_product >= 0.9999:
        return Quat.identQuat()

    rotation_axis = rest.cross(target)
    if rotation_axis.length_squared() <= EPSILON:
        fallback = Vec3(0, 1, 0)
        if abs(rest.dot(fallback)) > 0.95:
            fallback = Vec3(0, 0, 1)
        rotation_axis = rest.cross(fallback)
        if rotation_axis.length_squared() <= EPSILON:
            return Quat.identQuat()

    rotation_axis.normalize()
    rotation = Quat()
    rotation.setFromAxisAngleRad(math.acos(dot_product), rotation_axis)
    return rotation


def _basis_matrix(primary: Vec3, secondary: Vec3) -> LMatrix3f | None:
    x_axis = _normalized(primary)
    if x_axis is None:
        return None

    secondary_projected = secondary - (x_axis * secondary.dot(x_axis))
    y_axis = _normalized(secondary_projected)
    if y_axis is None:
        fallback = Vec3(0, 1, 0)
        if abs(x_axis.dot(fallback)) > 0.95:
            fallback = Vec3(0, 0, 1)
        secondary_projected = fallback - (x_axis * fallback.dot(x_axis))
        y_axis = _normalized(secondary_projected)
        if y_axis is None:
            return None

    z_axis = _normalized(x_axis.cross(y_axis))
    if z_axis is None:
        return None

    y_axis = _normalized(z_axis.cross(x_axis))
    if y_axis is None:
        return None

    basis = LMatrix3f()
    basis.setCol(0, x_axis)
    basis.setCol(1, y_axis)
    basis.setCol(2, z_axis)
    return basis


def _rotation_from_basis(
    rest_primary: Vec3,
    rest_secondary: Vec3,
    target_primary: Vec3,
    target_secondary: Vec3,
) -> Quat:
    rest_basis = _basis_matrix(rest_primary, rest_secondary)
    target_basis = _basis_matrix(target_primary, target_secondary)
    if rest_basis is None or target_basis is None:
        return Quat.identQuat()

    rest_basis_inverse = LMatrix3f(rest_basis)
    rest_basis_inverse.transposeInPlace()
    delta_matrix = target_basis * rest_basis_inverse

    rotation = Quat()
    rotation.setFromMatrix(delta_matrix)
    return rotation


def _build_basis(primary: Vec3, secondary: Vec3) -> tuple[Vec3, Vec3, Vec3] | None:
    basis_matrix = _basis_matrix(primary, secondary)
    if basis_matrix is None:
        return None
    return (
        _normalized(basis_matrix.getCol(0)),
        _normalized(basis_matrix.getCol(1)),
        _normalized(basis_matrix.getCol(2)),
    )


def _world_to_basis(vector: Vec3, basis: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    x_axis, y_axis, z_axis = basis
    return Vec3(vector.dot(x_axis), vector.dot(y_axis), vector.dot(z_axis))


def _basis_to_world(vector: Vec3, basis: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    x_axis, y_axis, z_axis = basis
    return (x_axis * vector.x) + (y_axis * vector.y) + (z_axis * vector.z)


def _apply_hand_curl_sign(vector: Vec3) -> Vec3:
    return Vec3(vector.x, vector.y, vector.z * HAND_CURL_SIGN)


def _align_direction_to_reference(direction: Vec3 | None, reference: Vec3 | None) -> Vec3 | None:
    aligned_direction = _normalized(direction)
    aligned_reference = _normalized(reference)
    if aligned_direction is None:
        return None
    if aligned_reference is None:
        return aligned_direction
    if aligned_direction.dot(aligned_reference) < 0.0:
        return aligned_direction * -1.0
    return aligned_direction


def _project_onto_plane(vector: Vec3 | None, plane_normal: Vec3 | None) -> Vec3 | None:
    candidate = _normalized(vector)
    normal = _normalized(plane_normal)
    if candidate is None:
        return None
    if normal is None:
        return candidate
    projected = candidate - (normal * candidate.dot(normal))
    return _normalized(projected)


def _constrain_finger_direction(
    target_direction: Vec3 | None,
    rest_direction: Vec3 | None,
    curl_axis: Vec3 | None,
) -> Vec3 | None:
    target = _normalized(target_direction)
    rest = _normalized(rest_direction)
    curl = _normalized(curl_axis)
    if target is None:
        return None
    if rest is None or curl is None:
        return _align_direction_to_reference(target, rest)

    plane_normal = _normalized(rest.cross(curl))
    if plane_normal is None:
        return _align_direction_to_reference(target, rest)

    constrained = _project_onto_plane(target, plane_normal)
    return _align_direction_to_reference(constrained, rest)


def _dampen_finger_sway(direction_in_basis: Vec3, segment_index: int) -> Vec3:
    sway_scale = FINGER_BASE_SWAY_SCALE if segment_index == 0 else FINGER_TIP_SWAY_SCALE
    return Vec3(direction_in_basis.x, direction_in_basis.y * sway_scale, direction_in_basis.z)


def _capture_finger_curl_angle(direction_in_basis: Vec3) -> float:
    forward_component = max(0.05, float(direction_in_basis.x))
    curl_angle = math.atan2(-float(direction_in_basis.z), forward_component)
    return _clamp(curl_angle, 0.0, MAX_FINGER_CURL_RADIANS)


class ASLLVDClip:
    def __init__(self, clip_path: str | Path) -> None:
        self.clip_path = Path(clip_path)
        with self.clip_path.open("rb") as handle:
            payload = pickle.load(handle)

        self.keypoints = payload.get("keypoints")
        self.confidences = payload.get("confidences")
        if self.keypoints is None or self.confidences is None:
            raise ValueError(f"ASLLVD clip is missing keypoints/confidences: {self.clip_path}")

        if len(self.keypoints.shape) != 3 or self.keypoints.shape[1] < TOTAL_LANDMARK_COUNT or self.keypoints.shape[2] != 3:
            raise ValueError(f"Unexpected ASLLVD keypoint shape {self.keypoints.shape} in {self.clip_path}")
        if len(self.confidences.shape) != 2 or self.confidences.shape[1] < TOTAL_LANDMARK_COUNT:
            raise ValueError(f"Unexpected ASLLVD confidence shape {self.confidences.shape} in {self.clip_path}")

        self.frame_count = int(self.keypoints.shape[0])
        self.available = self.frame_count > 0

    def frame_index_at_time(self, elapsed_seconds: float, fps: float) -> int:
        if not self.available:
            return 0
        return int(elapsed_seconds * fps) % self.frame_count

    def frame_data(self, frame_index: int):
        return self.keypoints[frame_index], self.confidences[frame_index]


class ASLLVDRigAnimator:
    def __init__(self, actor: Actor, config: AnimationConfig, fps: float = DEFAULT_FPS) -> None:
        self.actor = actor
        self.config = config
        self.backend = config.backend
        self.fps = fps

        if config.clip_path is None:
            raise ValueError("ASLLVD animation config must include a resolved clip path")

        self.clip = ASLLVDClip(config.clip_path)
        self.selected_clip_path = self.clip.clip_path
        self.selected_clip_source = f"{config.backend}:{config.gloss}:{self.selected_clip_path.name}"
        self.selected_gloss = config.gloss or self.selected_clip_path.stem

        self.world_joint_nodes: dict[str, object] = {}
        self.local_joint_nodes: dict[str, object] = {}
        self.rest_transforms: dict[str, JointRestTransform] = {}
        self.current_local_quats: dict[str, Quat] = {}
        self.controlled_joint_names: list[str] = []
        self._controlled_joint_name_set: set[str] = set()

        self.hand_control_joint_names = {side: f"FK-Hand.{side}" for side in SIDES}
        self.hand_parent_joint_names = {side: f"FK-Forearm.{side}" for side in SIDES}
        self.hand_root_joint_names = {side: f"DEF-Hand.{side}" for side in SIDES}
        self.arm_controls: dict[str, dict[str, ArmJointControl]] = {}
        self.finger_controls: dict[str, dict[str, tuple[FingerJointControl, ...]]] = {}
        self.arm_parent_world_quaternions: dict[str, Quat] = {}
        self.arm_rest_directions_world: dict[str, dict[str, Vec3]] = {}
        self.hand_rest_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.hand_rest_bases_local: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.hand_rest_bases_parent: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.previous_target_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.previous_arm_directions: dict[str, dict[str, Vec3]] = {}
        self.rest_torso_basis = self._build_rest_torso_basis()

        for side in SIDES:
            self.arm_controls[side] = self._build_arm_controls(side)
            self.finger_controls[side] = self._build_finger_controls(side)
            self.arm_parent_world_quaternions[side] = self._build_arm_parent_world_quat(side)
            self.arm_rest_directions_world[side] = self._build_arm_rest_directions(side)

            self.hand_rest_bases[side] = self._build_rest_hand_basis(side)
            hand_root_world_quat = _copy_quat(self._world_joint(self.hand_root_joint_names[side]).getQuat(self.actor))
            self.hand_rest_bases_local[side] = tuple(
                _normalized(hand_root_world_quat.conjugate().xform(axis)) or axis
                for axis in self.hand_rest_bases[side]
            )

            hand_joint_name = self.hand_control_joint_names[side]
            hand_rest_transform = self._joint_rest_transform(hand_joint_name)
            hand_world_quat = _copy_quat(self._world_joint(hand_joint_name).getQuat(self.actor))
            hand_parent_world_quat = hand_rest_transform.quat.conjugate() * hand_world_quat
            self.hand_rest_bases_parent[side] = tuple(
                _normalized(hand_parent_world_quat.conjugate().xform(axis)) or axis
                for axis in self.hand_rest_bases[side]
            )
            self.finger_controls[side] = self._configure_finger_controls(side, self.finger_controls[side])

            for control in self.arm_controls[side].values():
                self._register_joint(control.joint_name)
            self._register_joint(self.hand_control_joint_names[side])
            for controls in self.finger_controls[side].values():
                for control in controls:
                    self._register_joint(control.joint_name)

        for joint_name in self.controlled_joint_names:
            self.current_local_quats[joint_name] = _copy_quat(self.rest_transforms[joint_name].quat)

        self.enabled = self.clip.available
        self._last_frame_index: int | None = None
        self._reset_current_pose()
        self._apply_joint_quats(self.controlled_joint_names)
        self.actor.update()

    def _register_joint(self, joint_name: str) -> None:
        if joint_name in self._controlled_joint_name_set:
            return
        self._controlled_joint_name_set.add(joint_name)
        self.controlled_joint_names.append(joint_name)
        self._joint_rest_transform(joint_name)

    def _world_joint(self, joint_name: str):
        cached = self.world_joint_nodes.get(joint_name)
        if cached is not None:
            return cached
        joint = self.actor.exposeJoint(None, PART_NAME, joint_name)
        self.world_joint_nodes[joint_name] = joint
        return joint

    def _local_joint(self, joint_name: str):
        cached = self.local_joint_nodes.get(joint_name)
        if cached is not None:
            return cached
        joint = self.actor.exposeJoint(None, PART_NAME, joint_name, localTransform=1)
        self.local_joint_nodes[joint_name] = joint
        return joint

    def _joint_rest_transform(self, joint_name: str) -> JointRestTransform:
        cached = self.rest_transforms.get(joint_name)
        if cached is not None:
            return cached

        local_joint = self._local_joint(joint_name)
        transform = JointRestTransform(
            pos=_copy_vec(local_joint.getPos()),
            quat=_copy_quat(local_joint.getQuat()),
            scale=_copy_vec(local_joint.getScale()),
        )
        self.rest_transforms[joint_name] = transform
        return transform

    def _build_arm_parent_world_quat(self, side: str) -> Quat:
        upperarm_joint_name = f"FK-Upperarm.{side}"
        upperarm_rest = self._joint_rest_transform(upperarm_joint_name)
        upperarm_world_quat = _copy_quat(self._world_joint(upperarm_joint_name).getQuat(self.actor))
        return upperarm_rest.quat.conjugate() * upperarm_world_quat

    def _build_arm_rest_directions(self, side: str) -> dict[str, Vec3]:
        rest_directions: dict[str, Vec3] = {}
        for segment_name, (joint_base, child_base) in ARM_CONTROLLER_JOINTS.items():
            joint_name = f"{joint_base}.{side}"
            child_name = f"{child_base}.{side}"
            world_direction = _normalized(
                self._world_joint(child_name).getPos(self.actor) - self._world_joint(joint_name).getPos(self.actor)
            )
            rest_directions[segment_name] = world_direction or Vec3(1, 0, 0)
        return rest_directions

    def _build_rest_hand_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        hand_position = self._world_joint(f"DEF-Hand.{side}").getPos(self.actor)
        basis = _build_basis(
            self._world_joint(f"DEF-Middle1.{side}").getPos(self.actor) - hand_position,
            self._world_joint(f"DEF-Pinky1.{side}").getPos(self.actor) - self._world_joint(f"DEF-Index1.{side}").getPos(self.actor),
        )
        if basis is None:
            return (
                Vec3(1, 0, 0),
                Vec3(0, 1, 0),
                Vec3(0, 0, 1),
            )
        return basis

    def _build_rest_torso_basis(self) -> tuple[Vec3, Vec3, Vec3]:
        left_shoulder = self._world_joint("HNG-Upperarm_Parent.L").getPos(self.actor)
        right_shoulder = self._world_joint("HNG-Upperarm_Parent.R").getPos(self.actor)
        left_hip = self._world_joint("HNG-Thigh.L").getPos(self.actor)
        right_hip = self._world_joint("HNG-Thigh.R").getPos(self.actor)

        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        hip_center = (left_hip + right_hip) * 0.5
        basis = _build_basis(
            shoulder_center - hip_center,
            right_shoulder - left_shoulder,
        )
        if basis is None:
            return (
                Vec3(1, 0, 0),
                Vec3(0, 1, 0),
                Vec3(0, 0, 1),
            )
        return basis

    def _terminal_direction_local(self, joint_name: str, parent_name: str) -> Vec3:
        joint_world = self._world_joint(joint_name)
        parent_world = self._world_joint(parent_name)
        joint_world_quat = joint_world.getQuat(self.actor)
        world_direction = _normalized(joint_world.getPos(self.actor) - parent_world.getPos(self.actor))
        if world_direction is None:
            return Vec3(0, 0, 1)
        local_direction = _normalized(joint_world_quat.conjugate().xform(world_direction))
        return local_direction or Vec3(0, 0, 1)

    def _build_arm_controls(self, side: str) -> dict[str, ArmJointControl]:
        controls: dict[str, ArmJointControl] = {}
        for segment_name, (joint_base, child_base) in ARM_CONTROLLER_JOINTS.items():
            joint_name = f"{joint_base}.{side}"
            child_name = f"{child_base}.{side}"
            rest_transform = self._joint_rest_transform(joint_name)
            child_local = self._local_joint(child_name)
            rest_segment_direction = _normalized(child_local.getPos()) or Vec3(0, 0, 1)
            controls[segment_name] = ArmJointControl(
                joint_name=joint_name,
                rest_transform=rest_transform,
                rest_segment_direction_local=rest_segment_direction,
            )
        return controls

    def _build_finger_controls(self, side: str) -> dict[str, tuple[FingerJointControl, ...]]:
        controls: dict[str, tuple[FingerJointControl, ...]] = {}
        for finger_name, joint_bases in FINGER_CONTROLLER_JOINTS.items():
            chain_controls: list[FingerJointControl] = []
            joint_names = [f"{joint_base}.{side}" for joint_base in joint_bases]
            child_names = [
                f"{joint_base}.{side}" if joint_base is not None else None
                for joint_base in FINGER_CHILD_CONTROLLER_JOINTS[finger_name]
            ]

            for index, joint_name in enumerate(joint_names):
                rest_transform = self._joint_rest_transform(joint_name)
                child_name = child_names[index]
                if child_name is not None:
                    child_local = self._local_joint(child_name)
                    rest_segment_direction = _normalized(child_local.getPos()) or Vec3(0, 0, 1)
                else:
                    parent_name = joint_names[index - 1]
                    rest_segment_direction = self._terminal_direction_local(joint_name, parent_name)

                chain_controls.append(
                    FingerJointControl(
                        joint_name=joint_name,
                        rest_transform=rest_transform,
                        rest_segment_direction_local=rest_segment_direction,
                    )
                )

            controls[finger_name] = tuple(chain_controls)
        return controls

    def _detect_finger_curl_axis(self, side: str, control: FingerJointControl) -> tuple[Vec3, float]:
        parent_across_axis_world = self.hand_rest_bases[side][1]
        parent_palm_normal_world = self.hand_rest_bases[side][2]
        rest_direction_in_parent = _normalized(
            control.rest_transform.quat.xform(control.rest_segment_direction_local)
        )
        world_joint_quat = _copy_quat(self._world_joint(control.joint_name).getQuat(self.actor))
        parent_world_quat = control.rest_transform.quat.conjugate() * world_joint_quat
        parent_across_axis = _normalized(parent_world_quat.conjugate().xform(parent_across_axis_world))
        parent_palm_normal = _normalized(parent_world_quat.conjugate().xform(parent_palm_normal_world))
        if rest_direction_in_parent is None or parent_across_axis is None or parent_palm_normal is None:
            return Vec3(1, 0, 0), 1.0

        best_axis_local = Vec3(1, 0, 0)
        best_alignment = -1.0
        best_curl_sign = 1.0
        for axis_local in (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)):
            axis_in_parent = _normalized(control.rest_transform.quat.xform(axis_local))
            if axis_in_parent is None:
                continue

            alignment = abs(axis_in_parent.dot(parent_across_axis))
            if alignment <= best_alignment:
                continue

            derivative = _normalized(axis_in_parent.cross(rest_direction_in_parent))
            curl_sign = 1.0
            if derivative is not None and derivative.dot(parent_palm_normal) > 0.0:
                curl_sign = -1.0

            best_axis_local = axis_local
            best_alignment = alignment
            best_curl_sign = curl_sign

        return best_axis_local, best_curl_sign

    def _configure_finger_controls(
        self,
        side: str,
        controls_by_finger: dict[str, tuple[FingerJointControl, ...]],
    ) -> dict[str, tuple[FingerJointControl, ...]]:
        configured_controls: dict[str, tuple[FingerJointControl, ...]] = {}
        for finger_name, controls in controls_by_finger.items():
            updated_controls: list[FingerJointControl] = []
            for control in controls:
                curl_axis_local = control.curl_axis_local
                curl_sign = control.curl_sign
                if finger_name != "Thumb":
                    curl_axis_local, curl_sign = self._detect_finger_curl_axis(side, control)

                updated_controls.append(
                    FingerJointControl(
                        joint_name=control.joint_name,
                        rest_transform=control.rest_transform,
                        rest_segment_direction_local=control.rest_segment_direction_local,
                        curl_axis_local=curl_axis_local,
                        curl_sign=curl_sign,
                    )
                )
            configured_controls[finger_name] = tuple(updated_controls)
        return configured_controls

    def _reset_current_pose(self) -> None:
        for joint_name in self.controlled_joint_names:
            self.current_local_quats[joint_name] = _copy_quat(self.rest_transforms[joint_name].quat)

    def _apply_joint_quats(self, joint_names: list[str]) -> None:
        for joint_name in joint_names:
            rest_transform = self.rest_transforms[joint_name]
            self.actor.freezeJoint(
                PART_NAME,
                joint_name,
                transform=TransformState.makePosQuatScale(
                    rest_transform.pos,
                    self.current_local_quats[joint_name],
                    rest_transform.scale,
                ),
            )

    def _landmark_from_frame(self, frame_points, frame_confidences, index: int, minimum_confidence: float) -> Vec3 | None:
        if index >= len(frame_points) or index >= len(frame_confidences):
            return None
        if float(frame_confidences[index]) < minimum_confidence:
            return None
        point = frame_points[index]
        if abs(float(point[0])) <= EPSILON and abs(float(point[1])) <= EPSILON and abs(float(point[2])) <= EPSILON:
            return None
        return _mediapipe_to_rig_space(float(point[0]), float(point[1]), float(point[2]))

    def _capture_pose_landmarks(self, frame_points, frame_confidences) -> dict[int, Vec3]:
        pose_landmarks: dict[int, Vec3] = {}
        for index in (
            POSE_LEFT_SHOULDER,
            POSE_RIGHT_SHOULDER,
            POSE_LEFT_ELBOW,
            POSE_RIGHT_ELBOW,
            POSE_LEFT_WRIST,
            POSE_RIGHT_WRIST,
            POSE_LEFT_HIP,
            POSE_RIGHT_HIP,
        ):
            point = self._landmark_from_frame(frame_points, frame_confidences, index, MIN_POSE_CONFIDENCE)
            if point is not None:
                pose_landmarks[index] = point
        return pose_landmarks

    def _capture_hand_landmarks(self, frame_points, frame_confidences, offset: int) -> dict[int, Vec3] | None:
        hand_landmarks: dict[int, Vec3] = {}
        for index in range(HAND_LANDMARK_COUNT):
            point = self._landmark_from_frame(frame_points, frame_confidences, offset + index, MIN_HAND_CONFIDENCE)
            if point is not None:
                hand_landmarks[index] = point
        return hand_landmarks or None

    def _capture_frame_landmarks(self, frame_index: int) -> tuple[dict[int, Vec3], dict[str, dict[int, Vec3]]]:
        frame_points, frame_confidences = self.clip.frame_data(frame_index)
        pose_landmarks = self._capture_pose_landmarks(frame_points, frame_confidences)

        hands_by_side: dict[str, dict[int, Vec3]] = {}
        left_hand = self._capture_hand_landmarks(frame_points, frame_confidences, LEFT_HAND_OFFSET)
        right_hand = self._capture_hand_landmarks(frame_points, frame_confidences, RIGHT_HAND_OFFSET)
        if left_hand is not None:
            hands_by_side["L"] = left_hand
        if right_hand is not None:
            hands_by_side["R"] = right_hand
        return pose_landmarks, hands_by_side

    def _capture_torso_basis(self, pose_landmarks: dict[int, Vec3]) -> tuple[Vec3, Vec3, Vec3] | None:
        left_shoulder = pose_landmarks.get(POSE_LEFT_SHOULDER)
        right_shoulder = pose_landmarks.get(POSE_RIGHT_SHOULDER)
        left_hip = pose_landmarks.get(POSE_LEFT_HIP)
        right_hip = pose_landmarks.get(POSE_RIGHT_HIP)
        if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
            return None

        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        hip_center = (left_hip + right_hip) * 0.5
        return _build_basis(
            shoulder_center - hip_center,
            right_shoulder - left_shoulder,
        )

    def _remap_capture_direction(
        self,
        capture_direction: Vec3,
        capture_torso_basis: tuple[Vec3, Vec3, Vec3],
    ) -> Vec3 | None:
        torso_local_direction = _world_to_basis(capture_direction, capture_torso_basis)
        torso_local_direction.z *= TORSO_DEPTH_SIGN
        return _normalized(_basis_to_world(torso_local_direction, self.rest_torso_basis))

    def _update_arm_pose(self, side: str, pose_landmarks: dict[int, Vec3]) -> None:
        capture_torso_basis = self._capture_torso_basis(pose_landmarks)
        if capture_torso_basis is None:
            return

        shoulder_index = POSE_LEFT_SHOULDER if side == "L" else POSE_RIGHT_SHOULDER
        elbow_index = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        wrist_index = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        shoulder = pose_landmarks.get(shoulder_index)
        elbow = pose_landmarks.get(elbow_index)
        wrist = pose_landmarks.get(wrist_index)

        parent_world_quat = self.arm_parent_world_quaternions[side]

        upperarm_control = self.arm_controls[side]["Upperarm"]
        upperarm_local_quat = self.current_local_quats[upperarm_control.joint_name]
        upperarm_direction = _normalized(elbow - shoulder) if shoulder is not None and elbow is not None else None
        if upperarm_direction is not None:
            remapped_upperarm = self._remap_capture_direction(upperarm_direction, capture_torso_basis)
            if remapped_upperarm is not None:
                previous_upperarm = self.previous_arm_directions.get(side, {}).get("Upperarm")
                if previous_upperarm is not None:
                    remapped_upperarm = _blend_direction(previous_upperarm, remapped_upperarm, ARM_DIRECTION_BLEND) or remapped_upperarm
                target_direction_in_parent = _normalized(parent_world_quat.conjugate().xform(remapped_upperarm))
                rest_direction_in_parent = _normalized(
                    upperarm_control.rest_transform.quat.xform(upperarm_control.rest_segment_direction_local)
                )
                if target_direction_in_parent is not None and rest_direction_in_parent is not None:
                    upperarm_local_quat = upperarm_control.rest_transform.quat * _rotation_from_to(
                        rest_direction_in_parent,
                        target_direction_in_parent,
                    )
                    self.current_local_quats[upperarm_control.joint_name] = upperarm_local_quat
                    self.previous_arm_directions.setdefault(side, {})["Upperarm"] = remapped_upperarm
        parent_world_quat = upperarm_local_quat * parent_world_quat

        forearm_control = self.arm_controls[side]["Forearm"]
        forearm_direction = _normalized(wrist - elbow) if elbow is not None and wrist is not None else None
        if forearm_direction is not None:
            remapped_forearm = self._remap_capture_direction(forearm_direction, capture_torso_basis)
            if remapped_forearm is not None:
                previous_forearm = self.previous_arm_directions.get(side, {}).get("Forearm")
                if previous_forearm is not None:
                    remapped_forearm = _blend_direction(previous_forearm, remapped_forearm, ARM_DIRECTION_BLEND) or remapped_forearm
                target_direction_in_parent = _normalized(parent_world_quat.conjugate().xform(remapped_forearm))
                rest_direction_in_parent = _normalized(
                    forearm_control.rest_transform.quat.xform(forearm_control.rest_segment_direction_local)
                )
                if target_direction_in_parent is not None and rest_direction_in_parent is not None:
                    self.current_local_quats[forearm_control.joint_name] = forearm_control.rest_transform.quat * _rotation_from_to(
                        rest_direction_in_parent,
                        target_direction_in_parent,
                    )
                    self.previous_arm_directions.setdefault(side, {})["Forearm"] = remapped_forearm

    def _target_hand_basis(self, hand_landmarks: dict[int, Vec3]) -> tuple[Vec3, Vec3, Vec3] | None:
        wrist = hand_landmarks.get(0)
        index_mcp = hand_landmarks.get(5)
        middle_mcp = hand_landmarks.get(9)
        ring_mcp = hand_landmarks.get(13)
        pinky_mcp = hand_landmarks.get(17)
        palm_center = _average_vectors(index_mcp, middle_mcp, ring_mcp, pinky_mcp)
        if wrist is None or palm_center is None or index_mcp is None or pinky_mcp is None:
            return None

        palm_forward = _average_vectors(
            palm_center - wrist,
            index_mcp - wrist,
            pinky_mcp - wrist,
        )
        palm_across = _average_vectors(
            middle_mcp - index_mcp if middle_mcp is not None else None,
            ring_mcp - middle_mcp if ring_mcp is not None and middle_mcp is not None else None,
            pinky_mcp - ring_mcp if ring_mcp is not None else None,
            pinky_mcp - index_mcp,
        )
        if palm_forward is None or palm_across is None:
            return None
        return _build_basis(palm_forward, palm_across)

    def _stabilize_target_basis(
        self,
        side: str,
        target_basis: tuple[Vec3, Vec3, Vec3],
        previous_bases: dict[str, tuple[Vec3, Vec3, Vec3]],
    ) -> tuple[Vec3, Vec3, Vec3]:
        previous_basis = previous_bases.get(side)
        if previous_basis is None:
            previous_bases[side] = target_basis
            return target_basis

        x_axis, y_axis, z_axis = target_basis
        if previous_basis[2].dot(z_axis) < 0.0:
            y_axis *= -1.0
            z_axis *= -1.0

        blended_x = _normalized(_lerp_vec(previous_basis[0], x_axis, TARGET_BASIS_BLEND)) or x_axis
        blended_y = _normalized(_lerp_vec(previous_basis[1], y_axis, TARGET_BASIS_BLEND)) or y_axis
        stabilized_basis = _build_basis(blended_x, blended_y) or (x_axis, y_axis, z_axis)
        previous_bases[side] = stabilized_basis
        return stabilized_basis

    def _current_hand_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        hand_world_quat = _copy_quat(self._world_joint(self.hand_root_joint_names[side]).getQuat(self.actor))
        current_axes = tuple(
            _normalized(hand_world_quat.xform(axis_local)) or axis_world
            for axis_local, axis_world in zip(self.hand_rest_bases_local[side], self.hand_rest_bases[side])
        )
        return _build_basis(current_axes[0], current_axes[1]) or self.hand_rest_bases[side]

    def _update_finger_pose(
        self,
        side: str,
        finger_name: str,
        hand_landmarks: dict[int, Vec3],
        capture_hand_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        if finger_name != "Thumb":
            self._update_non_thumb_finger_pose(side, finger_name, hand_landmarks, capture_hand_basis)
            return

        current_hand_basis = self._current_hand_basis(side)
        hand_world_quat = _copy_quat(self._world_joint(self.hand_root_joint_names[side]).getQuat(self.actor))
        hand_curl_axis_world = current_hand_basis[2]

        target_segment_directions_world: list[Vec3] = []
        for segment_index, (start_index, end_index) in enumerate(FINGER_LANDMARKS[finger_name]):
            start_point = hand_landmarks.get(start_index)
            end_point = hand_landmarks.get(end_index)
            if start_point is None or end_point is None:
                return

            capture_world_direction = _normalized(end_point - start_point)
            if capture_world_direction is None:
                return

            direction_in_capture_basis = _apply_hand_curl_sign(_world_to_basis(capture_world_direction, capture_hand_basis))
            if finger_name != "Thumb":
                direction_in_capture_basis = _dampen_finger_sway(direction_in_capture_basis, segment_index)
            remapped_world_direction = _normalized(_basis_to_world(direction_in_capture_basis, current_hand_basis))
            if remapped_world_direction is None:
                return
            target_segment_directions_world.append(remapped_world_direction)

        parent_world_quat = hand_world_quat
        for control, target_world_direction in zip(self.finger_controls[side][finger_name], target_segment_directions_world):
            target_direction_in_parent = _normalized(parent_world_quat.conjugate().xform(target_world_direction))
            rest_direction_in_parent = _normalized(
                control.rest_transform.quat.xform(control.rest_segment_direction_local)
            )
            if target_direction_in_parent is None or rest_direction_in_parent is None:
                return

            parent_curl_axis = _normalized(parent_world_quat.conjugate().xform(hand_curl_axis_world))
            target_direction_in_parent = _constrain_finger_direction(
                target_direction_in_parent,
                rest_direction_in_parent,
                parent_curl_axis,
            )
            if target_direction_in_parent is None:
                return

            strength = THUMB_POSE_STRENGTH if finger_name == "Thumb" else FINGER_POSE_STRENGTH
            target_direction_in_parent = _blend_direction(
                rest_direction_in_parent,
                target_direction_in_parent,
                strength,
            ) or target_direction_in_parent

            desired_local_quat = control.rest_transform.quat * _rotation_from_to(
                rest_direction_in_parent,
                target_direction_in_parent,
            )
            self.current_local_quats[control.joint_name] = desired_local_quat
            parent_world_quat = desired_local_quat * parent_world_quat

    def _update_non_thumb_finger_pose(
        self,
        side: str,
        finger_name: str,
        hand_landmarks: dict[int, Vec3],
        capture_hand_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        for segment_index, ((start_index, end_index), control) in enumerate(
            zip(FINGER_LANDMARKS[finger_name], self.finger_controls[side][finger_name])
        ):
            start_point = hand_landmarks.get(start_index)
            end_point = hand_landmarks.get(end_index)
            if start_point is None or end_point is None:
                return

            capture_world_direction = _normalized(end_point - start_point)
            if capture_world_direction is None:
                return

            direction_in_capture_basis = _apply_hand_curl_sign(
                _world_to_basis(capture_world_direction, capture_hand_basis)
            )
            direction_in_capture_basis = _dampen_finger_sway(direction_in_capture_basis, segment_index)
            curl_angle = _capture_finger_curl_angle(direction_in_capture_basis) * FINGER_POSE_STRENGTH

            curl_axis_local = control.curl_axis_local or Vec3(1, 0, 0)
            curl_offset = Quat()
            curl_offset.setFromAxisAngleRad(curl_angle * control.curl_sign, curl_axis_local)
            self.current_local_quats[control.joint_name] = control.rest_transform.quat * curl_offset

    def update(self, task):
        if not self.enabled:
            return task.cont

        frame_index = self.clip.frame_index_at_time(task.time, self.fps)
        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            self.previous_target_bases.clear()
            self.previous_arm_directions.clear()
            self._reset_current_pose()
            self._apply_joint_quats(self.controlled_joint_names)
            self.actor.update()
        self._last_frame_index = frame_index

        pose_landmarks, hand_landmarks_by_side = self._capture_frame_landmarks(frame_index)
        capture_hand_bases_by_side: dict[str, tuple[Vec3, Vec3, Vec3]] = {}

        for side in SIDES:
            self._update_arm_pose(side, pose_landmarks)
        self._apply_joint_quats(
            [
                control.joint_name
                for side in SIDES
                for control in self.arm_controls[side].values()
            ]
        )
        self.actor.update()

        for side in SIDES:
            hand_landmarks = hand_landmarks_by_side.get(side)
            if hand_landmarks is None:
                self.previous_target_bases.pop(side, None)
                continue

            capture_hand_basis = self._target_hand_basis(hand_landmarks)
            if capture_hand_basis is None:
                self.previous_target_bases.pop(side, None)
                continue

            stabilized_capture_basis = self._stabilize_target_basis(side, capture_hand_basis, self.previous_target_bases)
            capture_hand_bases_by_side[side] = stabilized_capture_basis

        for side in SIDES:
            hand_landmarks = hand_landmarks_by_side.get(side)
            capture_hand_basis = capture_hand_bases_by_side.get(side)
            if hand_landmarks is None or capture_hand_basis is None:
                continue
            for finger_name in FINGER_LANDMARKS:
                self._update_finger_pose(side, finger_name, hand_landmarks, capture_hand_basis)

        self._apply_joint_quats(
            [
                control.joint_name
                for side in SIDES
                for controls in self.finger_controls[side].values()
                for control in controls
            ]
        )
        self.actor.update()
        return task.cont
