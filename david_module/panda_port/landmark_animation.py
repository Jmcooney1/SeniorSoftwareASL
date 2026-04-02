from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left

from direct.actor.Actor import Actor
from panda3d.core import LMatrix3f, Quat, TransformState, Vec3


PART_NAME = "modelRoot"
DEFAULT_FPS = 30.0
EPSILON = 1.0e-6
HAND_CURL_SIGN = -1.0
TEMPORAL_SMOOTHING_WEIGHTS = (0.15, 0.7, 0.15)
POSE_TEMPORAL_SMOOTHING_WEIGHTS = (0.2, 0.6, 0.2)
MAX_HAND_INTERPOLATION_GAP = 2
TARGET_BASIS_BLEND = 0.45
WRIST_FOREARM_BLEND = 0.35
WRIST_POSE_STRENGTH = 0.55
USE_POSE_WRIST = False
THUMB_CLIP_STRENGTH = 0.35
ARM_POSE_STRENGTH = 0.75
ARM_DIRECTION_BLEND = 0.3
TORSO_DEPTH_SIGN = -1.0

THUMB_BASELINE_LANDMARKS = ((0, 2), (1, 3), (2, 4))

ARM_CONTROLLER_JOINTS = {
    "Upperarm": ("FK-Upperarm", "FK-Forearm"),
    "Forearm": ("FK-Forearm", "FK-Hand"),
}

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

SIDES = ("L", "R")

FINGER_LANDMARKS = {
    "Thumb": ((1, 2), (2, 3), (3, 4)),
    "Index": ((5, 6), (6, 7), (7, 8)),
    "Middle": ((9, 10), (10, 11), (11, 12)),
    "Ring": ((13, 14), (14, 15), (15, 16)),
    "Pinky": ((17, 18), (18, 19), (19, 20)),
}

FINGER_CONTROLLER_JOINTS = {
    "Thumb": ("FK-Thumb1", "FK-Thumb2", "FK-Thumb3"),
    "Index": ("CARP-Index", "FK-Index1", "FK-Index2"),
    "Middle": ("CARP-Middle", "FK-Middle1", "FK-Middle2"),
    "Ring": ("CARP-Ring", "FK-Ring1", "FK-Ring2"),
    "Pinky": ("CARP-Pinky", "FK-Pinky1", "FK-Pinky2"),
}

FINGER_CHILD_CONTROLLER_JOINTS = {
    "Thumb": ("FK-Thumb2", "FK-Thumb3", None),
    "Index": ("FK-Index1", "FK-Index2", "FK-Index3"),
    "Middle": ("FK-Middle1", "FK-Middle2", "FK-Middle3"),
    "Ring": ("FK-Ring1", "FK-Ring2", "FK-Ring3"),
    "Pinky": ("FK-Pinky1", "FK-Pinky2", "FK-Pinky3"),
}


@dataclass(frozen=True)
class JointRestTransform:
    pos: Vec3
    quat: Quat
    scale: Vec3


@dataclass(frozen=True)
class FingerJointControl:
    joint_name: str
    rest_transform: JointRestTransform
    rest_segment_direction_local: Vec3


@dataclass(frozen=True)
class ArmJointControl:
    joint_name: str
    rest_transform: JointRestTransform
    rest_segment_direction_local: Vec3


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


class LandmarkClip:
    def __init__(self, pose_path: Path, hand_path: Path) -> None:
        self.pose_frames = self._smooth_pose_frames(self._load_pose_frames(pose_path))
        self.hand_frames = self._load_hand_frames(hand_path)
        hand_frames_by_side = self._assign_hands_to_sides(self.pose_frames, self.hand_frames)

        if hand_frames_by_side:
            self.start_frame = min(hand_frames_by_side)
            self.end_frame = max(hand_frames_by_side)
        elif self.hand_frames:
            self.start_frame = min(self.hand_frames)
            self.end_frame = max(self.hand_frames)
        elif self.pose_frames:
            self.start_frame = min(self.pose_frames)
            self.end_frame = max(self.pose_frames)
        else:
            self.start_frame = 0
            self.end_frame = -1

        self.frame_count = max(0, self.end_frame - self.start_frame + 1)
        self.hand_frames_by_side = self._densify_and_smooth_hand_frames_by_side(hand_frames_by_side)
        self.available = self.frame_count > 0 and bool(self.hand_frames_by_side)

    @staticmethod
    def _load_pose_frames(path: Path) -> dict[int, dict[int, Vec3]]:
        if not path.exists():
            return {}

        pose_frames: dict[int, dict[int, Vec3]] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                frame_number = int(row["frame"]) 
                landmark_index = int(row["landmark_index"])
                pose_frames.setdefault(frame_number, {})[landmark_index] = Vec3(
                    _mediapipe_to_rig_space(
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    )
                )
        return pose_frames

    @staticmethod
    def _load_hand_frames(path: Path) -> dict[int, dict[int, dict[int, Vec3]]]:
        if not path.exists():
            return {}

        hand_frames: dict[int, dict[int, dict[int, Vec3]]] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                frame_number = int(row["frame"]) 
                hand_index = int(row["hand_index"]) 
                landmark_index = int(row["landmark_index"]) 
                hand_frames.setdefault(frame_number, {}).setdefault(hand_index, {})[landmark_index] = (
                    _mediapipe_to_rig_space(
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    )
                )
        return hand_frames

    @classmethod
    def _smooth_pose_frames(cls, pose_frames: dict[int, dict[int, Vec3]]) -> dict[int, dict[int, Vec3]]:
        if not pose_frames:
            return {}

        smoothing_weights = POSE_TEMPORAL_SMOOTHING_WEIGHTS
        smoothed_pose_frames: dict[int, dict[int, Vec3]] = {}
        for frame_number in range(min(pose_frames), max(pose_frames) + 1):
            current_pose = pose_frames.get(frame_number)
            if current_pose is None:
                continue

            previous_pose = pose_frames.get(frame_number - 1, current_pose)
            next_pose = pose_frames.get(frame_number + 1, current_pose)
            smoothed_pose_frames[frame_number] = cls._weighted_average_landmarks(
                (
                    (previous_pose, smoothing_weights[0]),
                    (current_pose, smoothing_weights[1]),
                    (next_pose, smoothing_weights[2]),
                )
            )

        return smoothed_pose_frames

    @staticmethod
    def _weighted_average_landmarks(
        weighted_landmarks: tuple[tuple[dict[int, Vec3], float], ...],
    ) -> dict[int, Vec3]:
        averaged_landmarks: dict[int, Vec3] = {}
        landmark_indices = {
            landmark_index
            for landmarks, _ in weighted_landmarks
            for landmark_index in landmarks
        }

        for landmark_index in landmark_indices:
            total = Vec3(0, 0, 0)
            total_weight = 0.0
            for landmarks, weight in weighted_landmarks:
                landmark = landmarks.get(landmark_index)
                if landmark is None or weight <= 0.0:
                    continue
                total += landmark * weight
                total_weight += weight

            if total_weight > EPSILON:
                averaged_landmarks[landmark_index] = total * (1.0 / total_weight)

        return averaged_landmarks

    @classmethod
    def _interpolate_landmarks(
        cls,
        first_landmarks: dict[int, Vec3],
        second_landmarks: dict[int, Vec3],
        amount: float,
    ) -> dict[int, Vec3]:
        if amount <= 0.001:
            return {index: _copy_vec(point) for index, point in first_landmarks.items()}
        if amount >= 0.999:
            return {index: _copy_vec(point) for index, point in second_landmarks.items()}

        interpolated_landmarks: dict[int, Vec3] = {}
        landmark_indices = set(first_landmarks) | set(second_landmarks)
        for landmark_index in landmark_indices:
            first_point = first_landmarks.get(landmark_index)
            second_point = second_landmarks.get(landmark_index)
            if first_point is None:
                interpolated_landmarks[landmark_index] = _copy_vec(second_point)
            elif second_point is None:
                interpolated_landmarks[landmark_index] = _copy_vec(first_point)
            else:
                interpolated_landmarks[landmark_index] = _lerp_vec(first_point, second_point, amount)

        return interpolated_landmarks

    @classmethod
    def _frame_landmarks_for_side(
        cls,
        frame_number: int,
        side: str,
        assigned_frames: dict[int, dict[str, dict[int, Vec3]]],
        available_frames: list[int],
    ) -> dict[int, Vec3] | None:
        if not available_frames:
            return None

        direct_landmarks = assigned_frames.get(frame_number, {}).get(side)
        if direct_landmarks is not None:
            return {index: _copy_vec(point) for index, point in direct_landmarks.items()}

        if frame_number < available_frames[0] or frame_number > available_frames[-1]:
            return None

        insertion_index = bisect_left(available_frames, frame_number)
        if insertion_index <= 0 or insertion_index >= len(available_frames):
            return None

        previous_frame = available_frames[insertion_index - 1]
        next_frame = available_frames[insertion_index]
        gap_size = next_frame - previous_frame - 1
        if gap_size <= 0 or gap_size > MAX_HAND_INTERPOLATION_GAP:
            return None

        amount = _clamp((frame_number - previous_frame) / (next_frame - previous_frame), 0.0, 1.0)
        previous_landmarks = assigned_frames.get(previous_frame, {}).get(side, {})
        next_landmarks = assigned_frames.get(next_frame, {}).get(side, {})
        return cls._interpolate_landmarks(previous_landmarks, next_landmarks, amount)

    @classmethod
    def _densify_and_smooth_hand_frames_by_side(
        cls,
        assigned_frames: dict[int, dict[str, dict[int, Vec3]]],
    ) -> dict[int, dict[str, dict[int, Vec3]]]:
        if not assigned_frames:
            return {}

        start_frame = min(assigned_frames)
        end_frame = max(assigned_frames)
        frame_count = max(0, end_frame - start_frame + 1)
        if frame_count <= 0:
            return {}

        available_frames_by_side = {
            side: sorted(frame_number for frame_number, hands in assigned_frames.items() if side in hands)
            for side in SIDES
        }

        dense_frames: dict[int, dict[str, dict[int, Vec3]]] = {
            frame_number: {} for frame_number in range(start_frame, end_frame + 1)
        }
        for side in SIDES:
            available_frames = available_frames_by_side[side]
            if not available_frames:
                continue

            for frame_number in range(start_frame, end_frame + 1):
                frame_landmarks = cls._frame_landmarks_for_side(
                    frame_number,
                    side,
                    assigned_frames,
                    available_frames,
                )
                if frame_landmarks:
                    dense_frames[frame_number][side] = frame_landmarks

        smoothed_frames: dict[int, dict[str, dict[int, Vec3]]] = {
            frame_number: {} for frame_number in range(start_frame, end_frame + 1)
        }
        smoothing_weights = TEMPORAL_SMOOTHING_WEIGHTS
        for frame_number in range(start_frame, end_frame + 1):
            previous_frame = frame_number - 1
            next_frame = frame_number + 1
            for side in SIDES:
                current_landmarks = dense_frames[frame_number].get(side)
                if current_landmarks is None:
                    continue

                previous_landmarks = dense_frames.get(previous_frame, {}).get(side, current_landmarks)
                next_landmarks = dense_frames.get(next_frame, {}).get(side, current_landmarks)
                smoothed_frames[frame_number][side] = cls._weighted_average_landmarks(
                    (
                        (previous_landmarks, smoothing_weights[0]),
                        (current_landmarks, smoothing_weights[1]),
                        (next_landmarks, smoothing_weights[2]),
                    )
                )

        return smoothed_frames

    @staticmethod
    def _distance_squared(a: Vec3, b: Vec3) -> float:
        delta_x = a.x - b.x
        delta_y = a.y - b.y
        return (delta_x * delta_x) + (delta_y * delta_y)

    @classmethod
    def _assign_hands_to_sides(
        cls,
        pose_frames: dict[int, dict[int, Vec3]],
        hand_frames: dict[int, dict[int, dict[int, Vec3]]],
    ) -> dict[int, dict[str, dict[int, Vec3]]]:
        assigned_frames: dict[int, dict[str, dict[int, Vec3]]] = {}

        for frame_number, hands_in_frame in hand_frames.items():
            pose_landmarks = pose_frames.get(frame_number, {})
            left_wrist = pose_landmarks.get(POSE_LEFT_WRIST)
            right_wrist = pose_landmarks.get(POSE_RIGHT_WRIST)
            pose_available = left_wrist is not None and right_wrist is not None

            candidates = []
            for hand_index, hand_landmarks in hands_in_frame.items():
                wrist = hand_landmarks.get(0)
                if wrist is None:
                    continue
                candidates.append(
                    (
                        hand_index,
                        hand_landmarks,
                        cls._distance_squared(wrist, left_wrist),
                        cls._distance_squared(wrist, right_wrist),
                    )
                )

            if not candidates:
                continue

            frame_assignment: dict[str, dict[int, Vec3]] = {}
            if pose_available and len(candidates) == 1:
                _, hand_landmarks, left_cost, right_cost = candidates[0]
                frame_assignment["L" if left_cost <= right_cost else "R"] = hand_landmarks
            elif pose_available and len(candidates) > 1:
                first = candidates[0]
                second = candidates[1]
                direct_cost = first[2] + second[3]
                swapped_cost = first[3] + second[2]
                if direct_cost <= swapped_cost:
                    frame_assignment["L"] = first[1]
                    frame_assignment["R"] = second[1]
                else:
                    frame_assignment["L"] = second[1]
                    frame_assignment["R"] = first[1]
            elif len(candidates) == 1:
                _, hand_landmarks, _, _ = candidates[0]
                wrist = hand_landmarks.get(0)
                if wrist is not None:
                    frame_assignment["L" if wrist.x >= 0.5 else "R"] = hand_landmarks
            else:
                sorted_candidates = sorted(
                    candidates,
                    key=lambda candidate: candidate[1].get(0).x if candidate[1].get(0) is not None else float("-inf"),
                    reverse=True,
                )
                frame_assignment["L"] = sorted_candidates[0][1]
                frame_assignment["R"] = sorted_candidates[1][1]

            assigned_frames[frame_number] = frame_assignment

        return assigned_frames

    def frame_at_time(self, elapsed_seconds: float, fps: float) -> int:
        if not self.available:
            return 0
        frame_offset = int(elapsed_seconds * fps) % self.frame_count
        return self.start_frame + frame_offset

    def hands_at_time(self, elapsed_seconds: float, fps: float) -> dict[str, dict[int, Vec3]]:
        if not self.available:
            return {}

        frame_position = (elapsed_seconds * fps) % self.frame_count
        current_offset = int(math.floor(frame_position))
        next_offset = (current_offset + 1) % self.frame_count
        blend_amount = frame_position - current_offset

        current_frame = self.start_frame + current_offset
        next_frame = self.start_frame + next_offset
        current_hands = self.hand_frames_by_side.get(current_frame, {})
        next_hands = self.hand_frames_by_side.get(next_frame, {}) if next_frame > current_frame else {}

        sampled_hands: dict[str, dict[int, Vec3]] = {}
        for side in SIDES:
            current_landmarks = current_hands.get(side)
            next_landmarks = next_hands.get(side)
            if current_landmarks is None and next_landmarks is None:
                continue
            if current_landmarks is None:
                sampled_hands[side] = {index: _copy_vec(point) for index, point in next_landmarks.items()}
            elif next_landmarks is None:
                sampled_hands[side] = {index: _copy_vec(point) for index, point in current_landmarks.items()}
            else:
                sampled_hands[side] = self._interpolate_landmarks(current_landmarks, next_landmarks, blend_amount)

        return sampled_hands

    def pose_at_time(self, elapsed_seconds: float, fps: float) -> dict[int, Vec3]:
        if not self.pose_frames or self.frame_count <= 0:
            return {}

        frame_position = (elapsed_seconds * fps) % self.frame_count
        current_offset = int(math.floor(frame_position))
        next_offset = (current_offset + 1) % self.frame_count
        blend_amount = frame_position - current_offset

        current_frame = self.start_frame + current_offset
        next_frame = self.start_frame + next_offset
        current_pose = self.pose_frames.get(current_frame, {})
        next_pose = self.pose_frames.get(next_frame, {}) if next_frame > current_frame else {}

        if not current_pose and not next_pose:
            return {}
        if not current_pose:
            return {index: _copy_vec(point) for index, point in next_pose.items()}
        if not next_pose:
            return {index: _copy_vec(point) for index, point in current_pose.items()}
        return self._interpolate_landmarks(current_pose, next_pose, blend_amount)


class LandmarkRigAnimator:
    def __init__(
        self,
        actor: Actor,
        anim_dir: str | Path | None = None,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.actor = actor
        self.fps = fps
        self.anim_dir = Path(anim_dir) if anim_dir is not None else Path(__file__).resolve().parent / "anim"
        self.clip = LandmarkClip(
            self.anim_dir / "pose_output.csv",
            self.anim_dir / "hands_output.csv",
        )

        self.controlled_joint_names: list[str] = []
        self.rest_transforms: dict[str, JointRestTransform] = {}
        self.hand_rest_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.hand_rest_bases_local: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.hand_world_quaternions: dict[str, Quat] = {}
        self.hand_parent_world_quaternions: dict[str, Quat] = {}
        self.hand_rest_bases_parent: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.hand_control_joint_names = {side: f"FK-Hand.{side}" for side in SIDES}
        self.hand_root_joint_names = {side: f"DEF-Hand.{side}" for side in SIDES}
        self.arm_controls: dict[str, dict[str, ArmJointControl]] = {}
        self.arm_parent_world_quaternions: dict[str, Quat] = {}
        self.arm_rest_directions_world: dict[str, dict[str, Vec3]] = {}
        self.rest_torso_basis = self._build_rest_torso_basis()
        self.finger_controls: dict[str, dict[str, tuple[FingerJointControl, ...]]] = {}
        self.rest_finger_directions: dict[str, dict[str, tuple[Vec3, ...]]] = {}
        self.static_finger_directions: dict[str, dict[str, tuple[Vec3, ...]]] = {}
        self.previous_target_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.previous_wrist_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self.previous_arm_directions: dict[str, dict[str, Vec3]] = {}
        self.previous_frame_offset: int | None = None

        for side in SIDES:
            self.hand_rest_bases[side] = self._build_rest_hand_basis(side)
            hand_root_quat = _copy_quat(self.actor.exposeJoint(None, PART_NAME, self.hand_root_joint_names[side]).getQuat(self.actor))
            self.hand_rest_bases_local[side] = tuple(
                _normalized(hand_root_quat.conjugate().xform(axis)) or axis
                for axis in self.hand_rest_bases[side]
            )
            hand_joint_name = self.hand_control_joint_names[side]
            hand_rest_transform = self._joint_rest_transform(hand_joint_name)
            hand_world = self.actor.exposeJoint(None, PART_NAME, hand_joint_name)
            hand_world_quat = _copy_quat(hand_world.getQuat(self.actor))
            hand_parent_world_quat = hand_rest_transform.quat.conjugate() * hand_world_quat
            self.hand_world_quaternions[side] = hand_world_quat
            self.hand_parent_world_quaternions[side] = hand_parent_world_quat
            self.hand_rest_bases_parent[side] = tuple(
                _normalized(hand_parent_world_quat.conjugate().xform(axis)) or axis
                for axis in self.hand_rest_bases[side]
            )
            self.controlled_joint_names.append(hand_joint_name)
            self.arm_controls[side] = self._build_arm_controls(side)
            for arm_control in self.arm_controls[side].values():
                self.controlled_joint_names.append(arm_control.joint_name)
            self.finger_controls[side] = self._build_finger_controls(side)
            for controls in self.finger_controls[side].values():
                self.controlled_joint_names.extend(control.joint_name for control in controls)
            upperarm_world = _copy_quat(self.actor.exposeJoint(None, PART_NAME, f"FK-Upperarm.{side}").getQuat(self.actor))
            upperarm_rest = self.arm_controls[side]["Upperarm"].rest_transform
            self.arm_parent_world_quaternions[side] = upperarm_rest.quat.conjugate() * upperarm_world
            self.arm_rest_directions_world[side] = {
                segment_name: _normalized(
                    self.actor.exposeJoint(None, PART_NAME, f"{child_base}.{side}").getPos(self.actor)
                    - self.actor.exposeJoint(None, PART_NAME, f"{joint_base}.{side}").getPos(self.actor)
                ) or Vec3(1, 0, 0)
                for segment_name, (joint_base, child_base) in ARM_CONTROLLER_JOINTS.items()
            }
        self.rest_finger_directions = self._build_rest_finger_directions()
        self.static_finger_directions = self._build_static_finger_directions()

        self.enabled = self.clip.available

    def _joint_rest_transform(self, joint_name: str) -> JointRestTransform:
        cached = self.rest_transforms.get(joint_name)
        if cached is not None:
            return cached

        local_joint = self.actor.exposeJoint(None, PART_NAME, joint_name, localTransform=1)
        transform = JointRestTransform(
            pos=_copy_vec(local_joint.getPos()),
            quat=_copy_quat(local_joint.getQuat()),
            scale=_copy_vec(local_joint.getScale()),
        )
        self.rest_transforms[joint_name] = transform
        return transform

    def _build_rest_hand_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        hand_world = self.actor.exposeJoint(None, PART_NAME, f"DEF-Hand.{side}")
        index_world = self.actor.exposeJoint(None, PART_NAME, f"DEF-Index1.{side}")
        middle_world = self.actor.exposeJoint(None, PART_NAME, f"DEF-Middle1.{side}")
        pinky_world = self.actor.exposeJoint(None, PART_NAME, f"DEF-Pinky1.{side}")

        hand_position = hand_world.getPos(self.actor)
        basis = _build_basis(
            middle_world.getPos(self.actor) - hand_position,
            pinky_world.getPos(self.actor) - index_world.getPos(self.actor),
        )
        if basis is None:
            return (
                Vec3(1, 0, 0),
                Vec3(0, 1, 0),
                Vec3(0, 0, 1),
            )
        return basis

    def _build_rest_torso_basis(self) -> tuple[Vec3, Vec3, Vec3]:
        left_shoulder = self.actor.exposeJoint(None, PART_NAME, "HNG-Upperarm_Parent.L").getPos(self.actor)
        right_shoulder = self.actor.exposeJoint(None, PART_NAME, "HNG-Upperarm_Parent.R").getPos(self.actor)
        left_hip = self.actor.exposeJoint(None, PART_NAME, "HNG-Thigh.L").getPos(self.actor)
        right_hip = self.actor.exposeJoint(None, PART_NAME, "HNG-Thigh.R").getPos(self.actor)

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

    def _terminal_direction_local(self, side: str, joint_name: str, parent_name: str) -> Vec3:
        joint_world = self.actor.exposeJoint(None, PART_NAME, joint_name)
        parent_world = self.actor.exposeJoint(None, PART_NAME, parent_name)
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
            child_local = self.actor.exposeJoint(None, PART_NAME, child_name, localTransform=1)
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
                    child_local = self.actor.exposeJoint(None, PART_NAME, child_name, localTransform=1)
                    rest_segment_direction = _normalized(child_local.getPos()) or Vec3(0, 0, 1)
                else:
                    parent_name = joint_names[index - 1]
                    rest_segment_direction = self._terminal_direction_local(side, joint_name, parent_name)

                chain_controls.append(
                    FingerJointControl(
                        joint_name=joint_name,
                        rest_transform=rest_transform,
                        rest_segment_direction_local=rest_segment_direction,
                    )
                )

            controls[finger_name] = tuple(chain_controls)

        return controls

    def _build_rest_finger_directions(self) -> dict[str, dict[str, tuple[Vec3, ...]]]:
        rest_directions: dict[str, dict[str, tuple[Vec3, ...]]] = {side: {} for side in SIDES}

        for side in SIDES:
            hand_basis = self.hand_rest_bases[side]
            for finger_name, controls in self.finger_controls[side].items():
                finger_directions: list[Vec3] = []
                for control in controls:
                    joint = self.actor.exposeJoint(None, PART_NAME, control.joint_name)
                    world_direction = _normalized(joint.getQuat(self.actor).xform(control.rest_segment_direction_local))
                    if world_direction is None:
                        finger_directions = []
                        break

                    basis_direction = _normalized(_world_to_basis(world_direction, hand_basis))
                    if basis_direction is None:
                        finger_directions = []
                        break
                    finger_directions.append(basis_direction)

                if finger_directions:
                    rest_directions[side][finger_name] = tuple(finger_directions)

        return rest_directions

    def _build_static_finger_directions(self) -> dict[str, dict[str, tuple[Vec3, ...]]]:
        static_directions: dict[str, dict[str, tuple[Vec3, ...]]] = {side: {} for side in SIDES}

        for side in SIDES:
            collected: dict[str, list[list[Vec3]]] = {}
            for finger_name, segments in FINGER_LANDMARKS.items():
                baseline_segments = THUMB_BASELINE_LANDMARKS if finger_name == "Thumb" else segments
                collected[finger_name] = [[] for _ in baseline_segments]

            for frame_number in range(self.clip.start_frame, self.clip.end_frame + 1):
                hand_landmarks = self.clip.hand_frames_by_side.get(frame_number, {}).get(side)
                if hand_landmarks is None:
                    continue

                target_basis = self._target_hand_basis(hand_landmarks)
                if target_basis is None:
                    continue

                for finger_name, segment_samples in collected.items():
                    baseline_segments = THUMB_BASELINE_LANDMARKS if finger_name == "Thumb" else FINGER_LANDMARKS[finger_name]
                    for segment_index, (start_index, end_index) in enumerate(baseline_segments):
                        start_point = hand_landmarks.get(start_index)
                        end_point = hand_landmarks.get(end_index)
                        if start_point is None or end_point is None:
                            continue

                        world_direction = _normalized(end_point - start_point)
                        if world_direction is None:
                            continue

                        local_direction = _normalized(
                            _apply_hand_curl_sign(_world_to_basis(world_direction, target_basis))
                        )
                        if local_direction is None:
                            continue

                        if segment_samples[segment_index]:
                            reference_direction = segment_samples[segment_index][0]
                            if reference_direction.dot(local_direction) < 0.0:
                                local_direction *= -1.0
                        segment_samples[segment_index].append(local_direction)

            side_directions: dict[str, tuple[Vec3, ...]] = {}
            for finger_name, segment_samples in collected.items():
                averaged_segments: list[Vec3] = []
                for segment_index, samples in enumerate(segment_samples):
                    if not samples:
                        averaged_segments = []
                        break

                    average_direction = Vec3(0, 0, 0)
                    for sample in samples:
                        average_direction += sample
                    average_direction = _normalized(average_direction)
                    if average_direction is None:
                        averaged_segments = []
                        break

                    if finger_name == "Thumb":
                        rest_direction = self.rest_finger_directions.get(side, {}).get(finger_name, ())
                        if segment_index < len(rest_direction):
                            average_direction = _blend_direction(
                                rest_direction[segment_index],
                                average_direction,
                                THUMB_CLIP_STRENGTH,
                            ) or average_direction

                    averaged_segments.append(average_direction)

                if averaged_segments:
                    side_directions[finger_name] = tuple(averaged_segments)

            static_directions[side] = side_directions

        return static_directions

    def _reset_pose(self) -> None:
        for joint_name in self.controlled_joint_names:
            rest_transform = self.rest_transforms[joint_name]
            self.actor.freezeJoint(
                PART_NAME,
                joint_name,
                transform=TransformState.makePosQuatScale(
                    rest_transform.pos,
                    rest_transform.quat,
                    rest_transform.scale,
                ),
            )

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
            hand_landmarks.get(9) - index_mcp if hand_landmarks.get(9) is not None else None,
            hand_landmarks.get(13) - middle_mcp if hand_landmarks.get(13) is not None and middle_mcp is not None else None,
            pinky_mcp - ring_mcp if ring_mcp is not None else None,
            pinky_mcp - index_mcp,
        )
        if palm_forward is None or palm_across is None:
            return None

        return _build_basis(
            palm_forward,
            palm_across,
        )

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

    def _target_arm_directions(self, side: str, pose_landmarks: dict[int, Vec3]) -> dict[str, Vec3] | None:
        capture_torso_basis = self._capture_torso_basis(pose_landmarks)
        if capture_torso_basis is None:
            return None

        shoulder_index = POSE_LEFT_SHOULDER if side == "L" else POSE_RIGHT_SHOULDER
        elbow_index = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        wrist_index = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        shoulder = pose_landmarks.get(shoulder_index)
        elbow = pose_landmarks.get(elbow_index)
        wrist = pose_landmarks.get(wrist_index)
        if shoulder is None or elbow is None or wrist is None:
            return None

        capture_directions = {
            "Upperarm": _normalized(elbow - shoulder),
            "Forearm": _normalized(wrist - elbow),
        }
        if any(direction is None for direction in capture_directions.values()):
            return None

        previous_directions = self.previous_arm_directions.setdefault(side, {})
        target_directions: dict[str, Vec3] = {}
        for segment_name, capture_direction in capture_directions.items():
            torso_local_direction = _world_to_basis(capture_direction, capture_torso_basis)
            torso_local_direction.z *= TORSO_DEPTH_SIGN
            remapped_direction = _normalized(
                _basis_to_world(
                    torso_local_direction,
                    self.rest_torso_basis,
                )
            )
            if remapped_direction is None:
                return None

            target_direction = _blend_direction(
                self.arm_rest_directions_world[side][segment_name],
                remapped_direction,
                ARM_POSE_STRENGTH,
            ) or remapped_direction
            previous_direction = previous_directions.get(segment_name)
            if previous_direction is not None:
                target_direction = _blend_direction(
                    previous_direction,
                    target_direction,
                    ARM_DIRECTION_BLEND,
                ) or target_direction
            target_directions[segment_name] = target_direction

        previous_directions.update(target_directions)
        return target_directions

    def _apply_arm_pose(self, side: str, pose_landmarks: dict[int, Vec3]) -> bool:
        target_directions = self._target_arm_directions(side, pose_landmarks)
        if target_directions is None:
            return False

        parent_world_quat = self.arm_parent_world_quaternions[side]
        for segment_name in ("Upperarm", "Forearm"):
            control = self.arm_controls[side][segment_name]
            target_world_direction = target_directions.get(segment_name)
            if target_world_direction is None:
                return False

            target_direction_in_parent = _normalized(parent_world_quat.conjugate().xform(target_world_direction))
            rest_direction_in_parent = _normalized(
                control.rest_transform.quat.xform(control.rest_segment_direction_local)
            )
            if target_direction_in_parent is None or rest_direction_in_parent is None:
                desired_local_quat = _copy_quat(control.rest_transform.quat)
            else:
                delta_quat = _rotation_from_to(rest_direction_in_parent, target_direction_in_parent)
                desired_local_quat = control.rest_transform.quat * delta_quat

            self.actor.freezeJoint(
                PART_NAME,
                control.joint_name,
                transform=TransformState.makePosQuatScale(
                    control.rest_transform.pos,
                    desired_local_quat,
                    control.rest_transform.scale,
                ),
            )
            parent_world_quat = desired_local_quat * parent_world_quat

        return True

    def _current_hand_basis(
        self,
        side: str,
        hand_world_quat: Quat,
    ) -> tuple[Vec3, Vec3, Vec3]:
        current_axes = tuple(
            _normalized(hand_world_quat.xform(axis_local)) or axis_world
            for axis_local, axis_world in zip(self.hand_rest_bases_local[side], self.hand_rest_bases[side])
        )
        return _build_basis(current_axes[0], current_axes[1]) or self.hand_rest_bases[side]

    def _target_wrist_basis(
        self,
        side: str,
        hand_landmarks: dict[int, Vec3],
        pose_landmarks: dict[int, Vec3],
    ) -> tuple[Vec3, Vec3, Vec3] | None:
        capture_hand_basis = self._target_hand_basis(hand_landmarks)
        if capture_hand_basis is None:
            return None

        pose_wrist_index = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        pose_elbow_index = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        pose_wrist = pose_landmarks.get(pose_wrist_index)
        pose_elbow = pose_landmarks.get(pose_elbow_index)
        forearm_direction = _normalized(pose_wrist - pose_elbow) if pose_wrist is not None and pose_elbow is not None else None
        if forearm_direction is None:
            return None

        wrist_forward_capture = _blend_direction(
            capture_hand_basis[0],
            forearm_direction,
            WRIST_FOREARM_BLEND,
        ) or capture_hand_basis[0]
        wrist_basis_capture = _build_basis(wrist_forward_capture, capture_hand_basis[1]) or capture_hand_basis

        remapped_primary = _normalized(
            _basis_to_world(
                _world_to_basis(wrist_basis_capture[0], capture_hand_basis),
                self.hand_rest_bases[side],
            )
        )
        remapped_secondary = _normalized(
            _basis_to_world(
                _world_to_basis(wrist_basis_capture[1], capture_hand_basis),
                self.hand_rest_bases[side],
            )
        )
        if remapped_primary is None or remapped_secondary is None:
            return None

        return _build_basis(remapped_primary, remapped_secondary)

    def _apply_wrist_pose(
        self,
        side: str,
        hand_landmarks: dict[int, Vec3],
        pose_landmarks: dict[int, Vec3],
    ) -> None:
        target_basis = self._target_wrist_basis(side, hand_landmarks, pose_landmarks)
        if target_basis is None:
            return

        target_basis = self._stabilize_target_basis(side, target_basis, self.previous_wrist_bases)
        target_basis = _build_basis(
            _blend_direction(self.hand_rest_bases[side][0], target_basis[0], WRIST_POSE_STRENGTH) or target_basis[0],
            _blend_direction(self.hand_rest_bases[side][1], target_basis[1], WRIST_POSE_STRENGTH) or target_basis[1],
        ) or target_basis
        parent_world_quat = self.hand_parent_world_quaternions[side]
        rest_primary_parent, rest_secondary_parent, _ = self.hand_rest_bases_parent[side]
        target_primary_parent = _normalized(parent_world_quat.conjugate().xform(target_basis[0]))
        target_secondary_parent = _normalized(parent_world_quat.conjugate().xform(target_basis[1]))
        if target_primary_parent is None or target_secondary_parent is None:
            return

        delta_quat = _rotation_from_basis(
            rest_primary_parent,
            rest_secondary_parent,
            target_primary_parent,
            target_secondary_parent,
        )
        hand_joint_name = self.hand_control_joint_names[side]
        rest_transform = self.rest_transforms[hand_joint_name]
        desired_local_quat = rest_transform.quat * delta_quat
        self.actor.freezeJoint(
            PART_NAME,
            hand_joint_name,
            transform=TransformState.makePosQuatScale(
                rest_transform.pos,
                desired_local_quat,
                rest_transform.scale,
            ),
        )

    def _apply_hand_pose(self, side: str, hand_landmarks: dict[int, Vec3] | None) -> None:
        hand_world = self.actor.exposeJoint(None, PART_NAME, self.hand_root_joint_names[side])
        hand_world_quat = _copy_quat(hand_world.getQuat(self.actor))
        current_hand_basis = self._current_hand_basis(side, hand_world_quat)
        static_directions = self.static_finger_directions.get(side, {})
        target_basis = self._target_hand_basis(hand_landmarks) if hand_landmarks is not None else None
        if not static_directions and target_basis is None:
            return
        if target_basis is not None:
            target_basis = self._stabilize_target_basis(side, target_basis, self.previous_target_bases)

        for finger_name, segments in FINGER_LANDMARKS.items():
            target_segment_directions_world: list[Vec3] = []
            local_directions = static_directions.get(finger_name)
            if local_directions is not None and len(local_directions) == len(segments):
                for direction_in_target_basis in local_directions:
                    remapped_world_direction = _normalized(_basis_to_world(direction_in_target_basis, current_hand_basis))
                    if remapped_world_direction is None:
                        target_segment_directions_world = []
                        break
                    target_segment_directions_world.append(remapped_world_direction)
            else:
                for start_index, end_index in segments:
                    start_point = hand_landmarks.get(start_index)
                    end_point = hand_landmarks.get(end_index)
                    if start_point is None or end_point is None or target_basis is None:
                        target_segment_directions_world = []
                        break

                    target_world_direction = _normalized(end_point - start_point)
                    if target_world_direction is None:
                        target_segment_directions_world = []
                        break

                    direction_in_target_basis = _apply_hand_curl_sign(
                        _world_to_basis(target_world_direction, target_basis)
                    )
                    remapped_world_direction = _normalized(_basis_to_world(direction_in_target_basis, current_hand_basis))
                    if remapped_world_direction is None:
                        target_segment_directions_world = []
                        break
                    target_segment_directions_world.append(remapped_world_direction)

            if len(target_segment_directions_world) != len(segments):
                continue

            parent_world_quat = _copy_quat(hand_world_quat)
            for control, target_world_direction in zip(self.finger_controls[side][finger_name], target_segment_directions_world):
                target_direction_in_parent = _normalized(parent_world_quat.conjugate().xform(target_world_direction))
                rest_direction_in_parent = _normalized(
                    control.rest_transform.quat.xform(control.rest_segment_direction_local)
                )
                if target_direction_in_parent is None or rest_direction_in_parent is None:
                    desired_local_quat = _copy_quat(control.rest_transform.quat)
                else:
                    delta_quat = _rotation_from_to(rest_direction_in_parent, target_direction_in_parent)
                    desired_local_quat = control.rest_transform.quat * delta_quat

                self.actor.freezeJoint(
                    PART_NAME,
                    control.joint_name,
                    transform=TransformState.makePosQuatScale(
                        control.rest_transform.pos,
                        desired_local_quat,
                        control.rest_transform.scale,
                    ),
                )
                parent_world_quat = desired_local_quat * parent_world_quat

    def update(self, task):
        if not self.enabled:
            return task.cont

        frame_offset = int((task.time * self.fps) % self.clip.frame_count)
        if self.previous_frame_offset is not None and frame_offset < self.previous_frame_offset:
            self.previous_target_bases.clear()
            self.previous_wrist_bases.clear()
            self.previous_arm_directions.clear()
        self.previous_frame_offset = frame_offset

        hand_landmarks_by_side = self.clip.hands_at_time(task.time, self.fps)
        pose_landmarks = self.clip.pose_at_time(task.time, self.fps)
        self._reset_pose()
        for side in SIDES:
            if pose_landmarks and self._apply_arm_pose(side, pose_landmarks):
                continue
            self.previous_arm_directions.pop(side, None)

        self.actor.update()
        if USE_POSE_WRIST:
            for side in SIDES:
                hand_landmarks = hand_landmarks_by_side.get(side)
                if hand_landmarks is not None:
                    self._apply_wrist_pose(side, hand_landmarks, pose_landmarks)
                else:
                    self.previous_wrist_bases.pop(side, None)
                    self.previous_target_bases.pop(side, None)

            self.actor.update()
        for side in SIDES:
            hand_landmarks = hand_landmarks_by_side.get(side)
            if hand_landmarks is not None or self.static_finger_directions.get(side):
                self._apply_hand_pose(side, hand_landmarks)
            else:
                self.previous_arm_directions.pop(side, None)
                self.previous_wrist_bases.pop(side, None)
                self.previous_target_bases.pop(side, None)

        self.actor.update()
        return task.cont
