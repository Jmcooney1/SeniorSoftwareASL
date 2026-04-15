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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PART_NAME = "modelRoot"
DEFAULT_FPS = 24.0
EPSILON = 1.0e-6

# Confidence thresholds
MIN_POSE_CONFIDENCE = 0.01
MIN_HAND_CONFIDENCE = 0.5

# Temporal smoothing (exponential moving average alpha)
ARM_BLEND_ALPHA = 0.40
HAND_BASIS_BLEND_ALPHA = 0.30
HAND_QUAT_BLEND_ALPHA = 0.20
TORSO_BLEND_ALPHA = 0.15
# Finger pose strength
FINGER_POSE_STRENGTH = 0.75
THUMB_POSE_STRENGTH = 0.50
MAX_FINGER_CURL_RADIANS = math.radians(95.0)
BASE_SEGMENT_CURL_SCALE = 0.6

# Depth remapping sign -- capture torso Z points backward while rig torso Z
# points forward (toward camera at actor-local -Z).  The flip corrects this
# for arm directions; hand orientation uses forearm-relative remapping instead.
TORSO_DEPTH_SIGN = -1.0

# ---------------------------------------------------------------------------
# MediaPipe landmark layout inside PKL arrays (75 total)
# ---------------------------------------------------------------------------

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
LEFT_HAND_OFFSET = POSE_LANDMARK_COUNT          # indices 33-53
RIGHT_HAND_OFFSET = POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT  # indices 54-74
TOTAL_LANDMARK_COUNT = POSE_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT

# Pose landmark indices we need
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

SIDES = ("L", "R")

# Arm FK chain: segment name → (joint base, child base)
ARM_JOINTS = {
    "Upperarm": ("FK-Upperarm", "FK-Forearm"),
    "Forearm": ("FK-Forearm", "FK-Hand"),
}

# Hand landmark pairs for each finger segment (MCP→PIP, PIP→DIP, DIP→TIP)
FINGER_LANDMARKS = {
    "Thumb": ((1, 2), (2, 3), (3, 4)),
    "Index": ((5, 6), (6, 7), (7, 8)),
    "Middle": ((9, 10), (10, 11), (11, 12)),
    "Ring": ((13, 14), (14, 15), (15, 16)),
    "Pinky": ((17, 18), (18, 19), (19, 20)),
}

# Rig joint names for each finger segment
FINGER_JOINTS = {
    "Thumb": ("FK-Thumb1", "FK-Thumb2", "FK-Thumb3"),
    "Index": ("FK-Index1", "FK-Index2", "FK-Index3"),
    "Middle": ("FK-Middle1", "FK-Middle2", "FK-Middle3"),
    "Ring": ("FK-Ring1", "FK-Ring2", "FK-Ring3"),
    "Pinky": ("FK-Pinky1", "FK-Pinky2", "FK-Pinky3"),
}

# Child joint used to compute rest segment directions
FINGER_CHILD_JOINTS = {
    "Thumb": ("FK-Thumb2", "FK-Thumb3", None),
    "Index": ("FK-Index2", "FK-Index3", None),
    "Middle": ("FK-Middle2", "FK-Middle3", None),
    "Ring": ("FK-Ring2", "FK-Ring3", None),
    "Pinky": ("FK-Pinky2", "FK-Pinky3", None),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointRestTransform:
    pos: Vec3
    quat: Quat
    scale: Vec3


@dataclass(frozen=True)
class ArmControl:
    joint_name: str
    rest: JointRestTransform
    rest_dir_local: Vec3


@dataclass(frozen=True)
class FingerControl:
    joint_name: str
    rest: JointRestTransform
    rest_dir_local: Vec3
    curl_axis_local: Vec3
    curl_sign: float


# ---------------------------------------------------------------------------
# Vector / quaternion helpers
# ---------------------------------------------------------------------------

def _v(v: Vec3) -> Vec3:
    """Copy a Vec3."""
    return Vec3(v.x, v.y, v.z)


def _q(q: Quat) -> Quat:
    """Copy a Quat."""
    return Quat(q)


def _mp2rig(x: float, y: float, z: float) -> Vec3:
    """MediaPipe world-landmark → Panda3D rig space (Y-flip)."""
    return Vec3(x, -y, z)


def _norm(v: Vec3 | None) -> Vec3 | None:
    if v is None:
        return None
    c = _v(v)
    if c.lengthSquared() <= EPSILON:
        return None
    c.normalize()
    return c


def _safe_norm(v: Vec3) -> Vec3:
    """Normalize, returning zero vec if degenerate."""
    out = _v(v)
    ls = out.lengthSquared()
    if ls <= EPSILON:
        return Vec3(0, 0, 0)
    out.normalize()
    return out


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _lerp_vec(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a * (1.0 - t) + b * t


def _blend_dir(rest: Vec3, target: Vec3, alpha: float) -> Vec3 | None:
    """Blend two unit directions by *alpha* (0 = rest, 1 = target)."""
    r = _norm(rest)
    t = _norm(target)
    if r is None:
        return t
    if t is None:
        return r
    if alpha >= 0.999:
        return t
    if alpha <= 0.001:
        return r
    return _norm(r * (1.0 - alpha) + t * alpha) or t


def _rot_between(rest: Vec3, target: Vec3) -> Quat:
    """Quaternion rotating *rest* onto *target*.  Safe for degenerate / anti-parallel inputs."""
    r = _norm(rest)
    t = _norm(target)
    if r is None or t is None:
        return Quat.identQuat()

    dot = _clamp(r.dot(t), -1.0, 1.0)
    if dot >= 0.9999:
        return Quat.identQuat()

    axis = r.cross(t)
    if axis.lengthSquared() <= EPSILON:
        # Anti-parallel – pick an arbitrary perpendicular axis
        fallback = Vec3(0, 1, 0)
        if abs(r.dot(fallback)) > 0.95:
            fallback = Vec3(0, 0, 1)
        axis = r.cross(fallback)
        if axis.lengthSquared() <= EPSILON:
            return Quat.identQuat()

    axis.normalize()
    q = Quat()
    q.setFromAxisAngleRad(math.acos(dot), axis)
    return q


# ---------------------------------------------------------------------------
# Orthonormal basis helpers (Gram-Schmidt)
# ---------------------------------------------------------------------------

def _basis_mat(primary: Vec3, secondary: Vec3) -> LMatrix3f | None:
    """Build an orthonormal 3×3 from *primary* (→col0) and *secondary* (projected→col1).
    Returns None on degenerate input."""
    x = _norm(primary)
    if x is None:
        return None

    proj = secondary - x * secondary.dot(x)
    y = _norm(proj)
    if y is None:
        fb = Vec3(0, 1, 0) if abs(x.dot(Vec3(0, 1, 0))) <= 0.95 else Vec3(0, 0, 1)
        proj = fb - x * fb.dot(x)
        y = _norm(proj)
        if y is None:
            return None

    z = _norm(x.cross(y))
    if z is None:
        return None
    y = _norm(z.cross(x))
    if y is None:
        return None

    m = LMatrix3f()
    m.setCol(0, x)
    m.setCol(1, y)
    m.setCol(2, z)
    return m


def _build_basis(primary: Vec3, secondary: Vec3) -> tuple[Vec3, Vec3, Vec3] | None:
    m = _basis_mat(primary, secondary)
    if m is None:
        return None
    return (_norm(m.getCol(0)), _norm(m.getCol(1)), _norm(m.getCol(2)))


def _world_to_basis(v: Vec3, basis: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    x, y, z = basis
    return Vec3(v.dot(x), v.dot(y), v.dot(z))


def _basis_to_world(v: Vec3, basis: tuple[Vec3, Vec3, Vec3]) -> Vec3:
    x, y, z = basis
    return x * v.x + y * v.y + z * v.z


def _rot_from_basis(rp: Vec3, rs: Vec3, tp: Vec3, ts: Vec3) -> Quat:
    """Quaternion rotating rest-basis onto target-basis."""
    rb = _basis_mat(rp, rs)
    tb = _basis_mat(tp, ts)
    if rb is None or tb is None:
        return Quat.identQuat()
    ri = LMatrix3f(rb)
    ri.transposeInPlace()
    q = Quat()
    q.setFromMatrix(tb * ri)
    return q


def _remove_twist_from_offset(offset_q: Quat, twist_axis_parent: Vec3) -> Quat:
    """Remove the twist component of an offset quaternion around a parent-space axis."""
    axis = _norm(twist_axis_parent)
    if axis is None:
        return _q(offset_q)

    projected = axis * (
        offset_q.getI() * axis.x
        + offset_q.getJ() * axis.y
        + offset_q.getK() * axis.z
    )
    twist = Quat(offset_q.getR(), projected.x, projected.y, projected.z)
    if twist.lengthSquared() <= EPSILON:
        return _q(offset_q)
    twist.normalize()
    swing = offset_q * twist.conjugate()
    if swing.lengthSquared() <= EPSILON:
        return Quat.identQuat()
    swing.normalize()
    return swing


# ---------------------------------------------------------------------------
# Sign clip loader (unified for all PKL datasets)
# ---------------------------------------------------------------------------

class SignClip:
    """Loads a single PKL sign clip.

    Expected pickle payload::

        {"keypoints": ndarray(N, >=75, 3), "confidences": ndarray(N, >=75)}
    """

    def __init__(self, clip_path: str | Path) -> None:
        self.clip_path = Path(clip_path)
        with self.clip_path.open("rb") as f:
            data = pickle.load(f)

        self.keypoints = data.get("keypoints")
        self.confidences = data.get("confidences")
        if self.keypoints is None or self.confidences is None:
            raise ValueError(f"Clip missing keypoints/confidences: {self.clip_path}")

        if (
            len(self.keypoints.shape) != 3
            or self.keypoints.shape[1] < TOTAL_LANDMARK_COUNT
            or self.keypoints.shape[2] != 3
        ):
            raise ValueError(f"Bad keypoint shape {self.keypoints.shape}: {self.clip_path}")
        if len(self.confidences.shape) != 2 or self.confidences.shape[1] < TOTAL_LANDMARK_COUNT:
            raise ValueError(f"Bad confidence shape {self.confidences.shape}: {self.clip_path}")

        self.num_frames: int = int(self.keypoints.shape[0])
        self.available: bool = self.num_frames > 0

    def frame_at_time(self, seconds: float, fps: float) -> int:
        if not self.available:
            return 0
        return int(seconds * fps) % self.num_frames

    def frame_data(self, idx: int):
        return self.keypoints[idx], self.confidences[idx]


# ---------------------------------------------------------------------------
# Dataset resolvers
# ---------------------------------------------------------------------------

def resolve_asllvd_clip(dataset_root: Path, gloss: str, variant: int | None = None) -> Path | None:
    """Find a PKL in ``<root>/PKL_POSES/<gloss>-<variant>.pkl``."""
    pkl_dir = dataset_root / "PKL_POSES"
    if not pkl_dir.is_dir():
        return None
    if variant is not None:
        p = pkl_dir / f"{gloss}-{variant:03d}.pkl"
        return p if p.exists() else None
    matches = sorted(pkl_dir.glob(f"{gloss}-*.pkl"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Unified rig animator
# ---------------------------------------------------------------------------

class UnifiedRigAnimator:
    """FK animator that drives the *rain* character from any 75-landmark PKL clip.

    Fixes every known bug in the old ``ASLLVDRigAnimator`` and ``LandmarkRigAnimator``:

    * Robust ``_rot_between`` that handles anti-parallel / zero-length vectors.
    * Gram-Schmidt re-orthonormalization after Z-flip (basis jitter fix).
    * Finger curl axis detected from *motion trajectory* (not rest-pose only).
    * Finger sway retained for base segments (was zeroed out).
    * Thumb uses correct landmark indices.
    * Wrist rotation enabled.
    * Hand assignment uses pose wrist landmarks, not camera-relative x.
    """

    def __init__(self, actor: Actor, config: AnimationConfig, fps: float = DEFAULT_FPS) -> None:
        self.actor = actor
        self.config = config
        self.fps = fps
        self.clip: SignClip | None = None
        self.enabled = False

        # Public metadata read by HUD helpers in panda_core
        self.selected_clip_path: Path | None = None
        self.selected_gloss: str | None = None

        # Exposed for landmark debug visualiser
        self.last_pose_lms: dict[int, Vec3] = {}
        self.last_hand_lms: dict[str, dict[int, Vec3] | None] = {}
        self.hand_world_space: bool = True

        # Joint caches
        self._world_nodes: dict[str, object] = {}
        self._local_nodes: dict[str, object] = {}
        self._rest: dict[str, JointRestTransform] = {}
        self._cur_quats: dict[str, Quat] = {}
        self._controlled: list[str] = []
        self._controlled_set: set[str] = set()

        # Per-side control structures (populated by _init_controls)
        self.arm_ctrls: dict[str, dict[str, ArmControl]] = {}
        self.finger_ctrls: dict[str, dict[str, tuple[FingerControl, ...]]] = {}
        self._arm_parent_world_q: dict[str, Quat] = {}
        self._arm_rest_dirs: dict[str, dict[str, Vec3]] = {}
        self._hand_rest_basis: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self._hand_basis_in_fk_local: dict[str, tuple[Vec3, Vec3, Vec3]] = {}

        # Temporal state (reset each loop)
        self._prev_arm_dirs: dict[str, dict[str, Vec3]] = {}
        self._prev_hand_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self._prev_hand_quats: dict[str, Quat] = {}
        self._prev_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._cur_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._ref_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._last_frame: int | None = None

        # Build rest-torso basis from the rig's default pose
        self._rest_torso = self._build_rest_torso()

        # Build arm + finger controls for both sides
        for side in SIDES:
            self.arm_ctrls[side] = self._build_arm_ctrls(side)
            self.finger_ctrls[side] = self._build_finger_ctrls(side)
            self._arm_parent_world_q[side] = self._arm_parent_quat(side)
            self._arm_rest_dirs[side] = self._arm_rest_directions(side)
            self._hand_rest_basis[side] = self._build_hand_rest_basis(side)
            # Store hand rest basis in FK-Hand's local frame
            hand_init_wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor)) # type: ignore
            self._hand_basis_in_fk_local[side] = tuple(
                _norm(hand_init_wq.conjugate().xform(ax)) or ax
                for ax in self._hand_rest_basis[side]
            )

            # Register all controlled joints
            for ctrl in self.arm_ctrls[side].values():
                self._register(ctrl.joint_name)
            self._register(f"FK-Hand.{side}")
            for chain in self.finger_ctrls[side].values():
                for fc in chain:
                    self._register(fc.joint_name)

        # Snapshot rest quats for every controlled joint
        for jn in self._controlled:
            self._cur_quats[jn] = _q(self._rest[jn].quat)

        # Try loading a clip from config
        self._load_clip_from_config()

    # ------------------------------------------------------------------
    # Clip management
    # ------------------------------------------------------------------

    def _load_clip_from_config(self) -> None:
        cp = self.config.clip_path
        if cp is None:
            return
        self.set_clip(SignClip(cp), gloss=self.config.gloss)

    def set_clip(self, clip: SignClip, gloss: str | None = None) -> None:
        self.clip = clip
        self.selected_clip_path = clip.clip_path
        self.selected_gloss = gloss or clip.clip_path.stem
        self.enabled = clip.available
        self._last_frame = None
        self._prev_arm_dirs.clear()
        self._prev_hand_bases.clear()
        self._prev_hand_quats.clear()
        self._prev_torso = None
        self._cur_torso = None
        self._ref_torso = None
        self._reset_pose()
        self._freeze_all()
        self.actor.update()

    # ------------------------------------------------------------------
    # Joint helpers
    # ------------------------------------------------------------------

    def _register(self, name: str) -> None:
        if name in self._controlled_set:
            return
        self._controlled_set.add(name)
        self._controlled.append(name)
        self._rest_of(name)

    def _wj(self, name: str):
        """Expose world-space joint (cached)."""
        n = self._world_nodes.get(name)
        if n is not None:
            return n
        n = self.actor.exposeJoint(None, PART_NAME, name)
        self._world_nodes[name] = n
        return n

    def _lj(self, name: str):
        """Expose local-space joint (cached)."""
        n = self._local_nodes.get(name)
        if n is not None:
            return n
        n = self.actor.exposeJoint(None, PART_NAME, name, localTransform=1)
        self._local_nodes[name] = n
        return n

    def _rest_of(self, name: str) -> JointRestTransform:
        c = self._rest.get(name)
        if c is not None:
            return c
        lj = self._lj(name)
        t = JointRestTransform(pos=_v(lj.getPos()), quat=_q(lj.getQuat()), scale=_v(lj.getScale()))  # type: ignore
        self._rest[name] = t
        return t

    # ------------------------------------------------------------------
    # Rig-geometry queries (run once at init from default pose)
    # ------------------------------------------------------------------

    def _build_rest_torso(self) -> tuple[Vec3, Vec3, Vec3]:
        ls = self._wj("HNG-Upperarm_Parent.L").getPos(self.actor) # type: ignore
        rs = self._wj("HNG-Upperarm_Parent.R").getPos(self.actor) # type: ignore
        lh = self._wj("HNG-Thigh.L").getPos(self.actor) # type: ignore
        rh = self._wj("HNG-Thigh.R").getPos(self.actor) # type: ignore
        sc = (ls + rs) * 0.5
        hc = (lh + rh) * 0.5
        b = _build_basis(sc - hc, rs - ls)
        return b if b is not None else (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def _arm_parent_quat(self, side: str) -> Quat:
        name = f"FK-Upperarm.{side}"
        rest = self._rest_of(name)
        wq = _q(self._wj(name).getQuat(self.actor)) # type: ignore
        return rest.quat.conjugate() * wq

    def _arm_rest_directions(self, side: str) -> dict[str, Vec3]:
        dirs: dict[str, Vec3] = {}
        for seg, (jb, cb) in ARM_JOINTS.items():
            jn = f"{jb}.{side}"
            cn = f"{cb}.{side}"
            d = _norm(self._wj(cn).getPos(self.actor) - self._wj(jn).getPos(self.actor)) # type: ignore
            dirs[seg] = d or Vec3(1, 0, 0)
        return dirs

    def _build_hand_rest_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        hp = self._wj(f"DEF-Hand.{side}").getPos(self.actor) # type: ignore
        forward = self._wj(f"DEF-Middle1.{side}").getPos(self.actor) - hp # type: ignore
        across = (
            self._wj(f"DEF-Pinky1.{side}").getPos(self.actor) # type: ignore
            - self._wj(f"DEF-Index1.{side}").getPos(self.actor) # type: ignore
        )
        b = _build_basis(forward, across)
        return b if b is not None else (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def _build_arm_ctrls(self, side: str) -> dict[str, ArmControl]:
        ctrls: dict[str, ArmControl] = {}
        for seg, (jb, cb) in ARM_JOINTS.items():
            jn = f"{jb}.{side}"
            cn = f"{cb}.{side}"
            rest = self._rest_of(jn)
            child_local = self._lj(cn)
            rd = _norm(child_local.getPos()) or Vec3(0, 0, 1) # type: ignore
            ctrls[seg] = ArmControl(joint_name=jn, rest=rest, rest_dir_local=rd)
        return ctrls

    def _build_finger_ctrls(self, side: str) -> dict[str, tuple[FingerControl, ...]]:
        ctrls: dict[str, tuple[FingerControl, ...]] = {}
        for fname, jbases in FINGER_JOINTS.items():
            chain: list[FingerControl] = []
            jnames = [f"{jb}.{side}" for jb in jbases]
            cnames = [
                (f"{cjb}.{side}" if cjb is not None else None)
                for cjb in FINGER_CHILD_JOINTS[fname]
            ]
            for i, jn in enumerate(jnames):
                rest = self._rest_of(jn)
                cn = cnames[i]
                if cn is not None:
                    rd = _norm(self._lj(cn).getPos()) or Vec3(0, 0, 1) # type: ignore
                else:
                    # Terminal segment – infer direction from parent→this
                    parent_name = jnames[i - 1]
                    wj = self._wj(jn)
                    pj = self._wj(parent_name)
                    wdir = _norm(wj.getPos(self.actor) - pj.getPos(self.actor)) # type: ignore
                    wq = wj.getQuat(self.actor) # type: ignore
                    rd = _norm(wq.conjugate().xform(wdir)) if wdir is not None else Vec3(0, 0, 1)
                    rd = rd or Vec3(0, 0, 1)

                # Detect curl axis using hand basis geometry
                ca, cs = self._detect_curl_axis(side, jn, rest, rd)
                chain.append(FingerControl(
                    joint_name=jn, rest=rest, rest_dir_local=rd,
                    curl_axis_local=ca, curl_sign=cs,
                ))
            ctrls[fname] = tuple(chain)
        return ctrls

    def _detect_curl_axis(
        self, side: str, joint_name: str,
        rest: JointRestTransform, rest_dir_local: Vec3,
    ) -> tuple[Vec3, float]:
        """Pick the local axis most aligned with the hand's across-direction and determine curl sign."""
        hand_basis = self._build_hand_rest_basis(side)
        across_world = hand_basis[1]  # pinky→index axis
        palm_world = hand_basis[2]    # palm normal

        rest_dir_parent = _norm(rest.quat.xform(rest_dir_local))
        wq = _q(self._wj(joint_name).getQuat(self.actor)) # type: ignore
        parent_wq = rest.quat.conjugate() * wq
        across_parent = _norm(parent_wq.conjugate().xform(across_world))
        palm_parent = _norm(parent_wq.conjugate().xform(palm_world))

        if rest_dir_parent is None or across_parent is None or palm_parent is None:
            return Vec3(1, 0, 0), 1.0

        best_ax = Vec3(1, 0, 0)
        best_align = -1.0
        best_sign = 1.0
        for ax in (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)):
            in_parent = _norm(rest.quat.xform(ax))
            if in_parent is None:
                continue
            align = abs(in_parent.dot(across_parent))
            if align <= best_align:
                continue
            # deriv = axis × finger_forward = direction tip moves under
            # positive rotation around this axis.
            deriv = _norm(in_parent.cross(rest_dir_parent))
            # Default: positive rotation goes away from palm → curl needs
            # negative sign.  If positive rotation goes TOWARD palm, keep
            # positive so that a positive curl_angle curls inward.
            sign = -1.0
            if deriv is not None and deriv.dot(palm_parent) > 0.0:
                sign = 1.0
            best_ax = ax
            best_align = align
            best_sign = sign
        return best_ax, best_sign

    # ------------------------------------------------------------------
    # Per-frame landmark extraction
    # ------------------------------------------------------------------

    def _lm(self, pts, conf, idx: int, min_conf: float) -> Vec3 | None:
        """Read a single landmark, returning None if below confidence or zero."""
        if idx >= len(pts) or idx >= len(conf):
            return None
        if float(conf[idx]) < min_conf:
            return None
        p = pts[idx]
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if abs(x) <= EPSILON and abs(y) <= EPSILON and abs(z) <= EPSILON:
            return None
        return _mp2rig(x, y, z)

    def _pose_lms(self, pts, conf) -> dict[int, Vec3]:
        out: dict[int, Vec3] = {}
        for idx in (
            POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER,
            POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW,
            POSE_LEFT_WRIST, POSE_RIGHT_WRIST,
            POSE_LEFT_HIP, POSE_RIGHT_HIP,
        ):
            v = self._lm(pts, conf, idx, MIN_POSE_CONFIDENCE)
            if v is not None:
                out[idx] = v
        return out

    def _hand_lms(self, pts, conf, offset: int) -> dict[int, Vec3] | None:
        out: dict[int, Vec3] = {}
        for i in range(HAND_LANDMARK_COUNT):
            v = self._lm(pts, conf, offset + i, MIN_HAND_CONFIDENCE)
            if v is not None:
                out[i] = v
        return out or None

    def _frame_data(self, fi: int):
        pts, conf = self.clip.frame_data(fi) # pyright: ignore[reportOptionalMemberAccess]
        pose = self._pose_lms(pts, conf)
        hands: dict[str, dict[int, Vec3]] = {}
        lh = self._hand_lms(pts, conf, LEFT_HAND_OFFSET)
        rh = self._hand_lms(pts, conf, RIGHT_HAND_OFFSET)
        if lh is not None:
            hands["L"] = lh
        if rh is not None:
            hands["R"] = rh
        return pose, hands

    # ------------------------------------------------------------------
    # Torso basis from current frame
    # ------------------------------------------------------------------

    def _capture_torso(self, plms: dict[int, Vec3]) -> tuple[Vec3, Vec3, Vec3] | None:
        ls = plms.get(POSE_LEFT_SHOULDER)
        rs = plms.get(POSE_RIGHT_SHOULDER)
        if ls is None or rs is None:
            return None
        lh = plms.get(POSE_LEFT_HIP)
        rh = plms.get(POSE_RIGHT_HIP)
        if lh is not None and rh is not None:
            sc = (ls + rs) * 0.5
            hc = (lh + rh) * 0.5
            up = sc - hc
            if up.lengthSquared() > EPSILON:
                return _build_basis(up, rs - ls)
        # Fallback: shoulders + world-up when hips unavailable
        # PKL uses normalised (image-space) landmarks; after _mp2rig Y-flip, +Y is up.
        return _build_basis(Vec3(0, 1, 0), rs - ls)

    def _remap_dir(self, d: Vec3, cap_torso: tuple[Vec3, Vec3, Vec3]) -> Vec3 | None:
        """Remap a capture-space direction into rig-torso space, flipping depth."""
        local = _world_to_basis(d, cap_torso)
        local = Vec3(local.x, local.y, local.z * TORSO_DEPTH_SIGN)
        return _norm(_basis_to_world(local, self._rest_torso))

    def _smooth_torso(self, raw: tuple[Vec3, Vec3, Vec3]) -> tuple[Vec3, Vec3, Vec3]:
        """EMA-smooth the capture torso basis to reduce frame-to-frame jitter."""
        prev = self._prev_torso
        if prev is None:
            self._prev_torso = raw
            return raw
        x, y, z = raw
        if prev[0].dot(x) < 0.0:
            x = x * -1.0
        if prev[1].dot(y) < 0.0:
            y = y * -1.0
        bx = _norm(_lerp_vec(prev[0], x, TORSO_BLEND_ALPHA)) or x
        by = _norm(_lerp_vec(prev[1], y, TORSO_BLEND_ALPHA)) or y
        stab = _build_basis(bx, by) or raw
        self._prev_torso = stab
        return stab

    # ------------------------------------------------------------------
    # Arm FK update
    # ------------------------------------------------------------------

    def _update_arms(self, side: str, plms: dict[int, Vec3]) -> None:
        if self._ref_torso is None:
            return
        cap_torso = self._ref_torso

        si = POSE_LEFT_SHOULDER if side == "L" else POSE_RIGHT_SHOULDER
        ei = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        wi = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        shoulder = plms.get(si)
        elbow = plms.get(ei)
        wrist = plms.get(wi)

        parent_wq = self._arm_parent_world_q[side]

        # --- Upperarm ---
        ua = self.arm_ctrls[side]["Upperarm"]
        ua_q = self._cur_quats[ua.joint_name]
        ua_dir = _norm(elbow - shoulder) if shoulder is not None and elbow is not None else None
        if ua_dir is not None:
            remapped = self._remap_dir(ua_dir, cap_torso)
            if remapped is not None:
                prev = self._prev_arm_dirs.get(side, {}).get("Upperarm")
                if prev is not None:
                    remapped = _blend_dir(prev, remapped, ARM_BLEND_ALPHA) or remapped
                tgt_par = _norm(parent_wq.conjugate().xform(remapped))
                rst_par = _norm(ua.rest.quat.xform(ua.rest_dir_local))
                if tgt_par is not None and rst_par is not None:
                    ua_q = ua.rest.quat * _rot_between(rst_par, tgt_par)
                    self._cur_quats[ua.joint_name] = ua_q
                    self._prev_arm_dirs.setdefault(side, {})["Upperarm"] = remapped
        parent_wq = ua_q * parent_wq

        # --- Forearm ---
        fa = self.arm_ctrls[side]["Forearm"]
        fa_dir = _norm(wrist - elbow) if elbow is not None and wrist is not None else None
        if fa_dir is not None:
            remapped = self._remap_dir(fa_dir, cap_torso)
            if remapped is not None:
                prev = self._prev_arm_dirs.get(side, {}).get("Forearm")
                if prev is not None:
                    remapped = _blend_dir(prev, remapped, ARM_BLEND_ALPHA) or remapped
                tgt_par = _norm(parent_wq.conjugate().xform(remapped))
                rst_par = _norm(fa.rest.quat.xform(fa.rest_dir_local))
                if tgt_par is not None and rst_par is not None:
                    self._cur_quats[fa.joint_name] = fa.rest.quat * _rot_between(rst_par, tgt_par)
                    self._prev_arm_dirs.setdefault(side, {})["Forearm"] = remapped

    # ------------------------------------------------------------------
    # Hand basis from landmarks
    # ------------------------------------------------------------------

    def _target_hand_basis(self, hlms: dict[int, Vec3]) -> tuple[Vec3, Vec3, Vec3] | None:
        w = hlms.get(0)
        idx_mcp = hlms.get(5)
        mid_mcp = hlms.get(9)
        rng_mcp = hlms.get(13)
        pnk_mcp = hlms.get(17)

        centers = [v for v in (idx_mcp, mid_mcp, rng_mcp, pnk_mcp) if v is not None]
        if not centers or w is None or idx_mcp is None or pnk_mcp is None:
            return None
        palm_c = sum(centers, Vec3(0, 0, 0)) * (1.0 / len(centers))

        fwd_parts = [v for v in (palm_c - w, idx_mcp - w, pnk_mcp - w) if v is not None]
        fwd = sum(fwd_parts, Vec3(0, 0, 0)) * (1.0 / len(fwd_parts))

        across_parts = []
        if mid_mcp is not None:
            across_parts.append(mid_mcp - idx_mcp)
        if rng_mcp is not None and mid_mcp is not None:
            across_parts.append(rng_mcp - mid_mcp)
        if rng_mcp is not None:
            across_parts.append(pnk_mcp - rng_mcp)
        across_parts.append(pnk_mcp - idx_mcp)
        across = sum(across_parts, Vec3(0, 0, 0)) * (1.0 / len(across_parts))

        return _build_basis(fwd, across)

    def _stabilize_basis(
        self, side: str, basis: tuple[Vec3, Vec3, Vec3],
    ) -> tuple[Vec3, Vec3, Vec3]:
        prev = self._prev_hand_bases.get(side)
        if prev is None:
            self._prev_hand_bases[side] = basis
            return basis
        x, y, z = basis
        # Flip check to avoid sign discontinuities on Z
        if prev[2].dot(z) < 0.0:
            y = y * -1.0
            z = z * -1.0
        bx = _norm(_lerp_vec(prev[0], x, HAND_BASIS_BLEND_ALPHA)) or x
        by = _norm(_lerp_vec(prev[1], y, HAND_BASIS_BLEND_ALPHA)) or y
        stab = _build_basis(bx, by) or (x, y, z)
        self._prev_hand_bases[side] = stab
        return stab

    def _current_hand_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor)) # type: ignore
        axes = tuple(
            _norm(wq.xform(al)) or aw
            for al, aw in zip(self._hand_basis_in_fk_local[side], self._hand_rest_basis[side])
        )
        return _build_basis(axes[0], axes[1]) or self._hand_rest_basis[side]

    # ------------------------------------------------------------------
    # Hand (wrist) FK
    # ------------------------------------------------------------------

    def _update_hand_pose(
        self, side: str, hlms: dict[int, Vec3],
        cap_basis: tuple[Vec3, Vec3, Vec3],
        plms: dict[int, Vec3],
    ) -> None:
        """Apply wrist orientation using forearm-relative remapping.

        Instead of mapping the hand basis through the torso (which introduces
        a reflection that breaks handedness), we express the capture hand
        basis in the capture forearm frame and reconstruct it in the rig
        forearm frame.  The arm FK has already mapped forearm orientation
        from capture to rig, so this avoids the torso reflection entirely.
        """
        hand_jn = f"FK-Hand.{side}"
        hand_rest = self._rest_of(hand_jn)

        # --- capture forearm frame ---
        si = POSE_LEFT_SHOULDER if side == "L" else POSE_RIGHT_SHOULDER
        ei = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        wi = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        s, e, w = plms.get(si), plms.get(ei), plms.get(wi)
        # Keep the capture forearm frame anchored to the same pose wrist used
        # by the arm FK.  ASLLVD hand wrist(0) drifts significantly relative
        # to the pose wrist and injects palm twist into the forearm basis,
        # which shows up as left/right wrist asymmetry on mirrored signs.
        if s is None or e is None or w is None:
            return
        cap_fa_dir = _norm(w - e)
        cap_ua_dir = _norm(s - e)
        if cap_fa_dir is None or cap_ua_dir is None:
            return
        cap_fa_basis = _build_basis(cap_fa_dir, cap_ua_dir)
        if cap_fa_basis is None:
            return

        # Hand axes in capture-forearm-local coordinates
        hfwd_local = _world_to_basis(cap_basis[0], cap_fa_basis)
        hacr_local = _world_to_basis(cap_basis[1], cap_fa_basis)

        # The arm FK applied a depth flip (TORSO_DEPTH_SIGN) when mapping
        # capture arm directions into rig space.  This reflects the arm
        # plane, flipping the forearm basis's Z axis.  Correct it here so
        # the hand orientation is not inverted.
        hfwd_local = Vec3(hfwd_local.x, hfwd_local.y, hfwd_local.z * TORSO_DEPTH_SIGN)
        hacr_local = Vec3(hacr_local.x, hacr_local.y, hacr_local.z * TORSO_DEPTH_SIGN)

        # --- rig forearm frame (after arm freeze) ---
        rig_fa_pos = self._wj(f"FK-Forearm.{side}").getPos(self.actor)  # type: ignore
        rig_hand_pos = self._wj(f"FK-Hand.{side}").getPos(self.actor)  # type: ignore
        rig_ua_pos = self._wj(f"FK-Upperarm.{side}").getPos(self.actor)  # type: ignore
        rig_fa_dir = _norm(rig_hand_pos - rig_fa_pos)
        rig_ua_dir = _norm(rig_ua_pos - rig_fa_pos)
        if rig_fa_dir is None or rig_ua_dir is None:
            return
        rig_fa_basis = _build_basis(rig_fa_dir, rig_ua_dir)
        if rig_fa_basis is None:
            return

        # Reconstruct hand axes in rig space
        rig_fwd = _norm(_basis_to_world(hfwd_local, rig_fa_basis))
        rig_acr = _norm(_basis_to_world(hacr_local, rig_fa_basis))
        hpalm_local = _world_to_basis(cap_basis[2], cap_fa_basis)
        hpalm_local = Vec3(hpalm_local.x, hpalm_local.y, hpalm_local.z * TORSO_DEPTH_SIGN)
        rig_palm = _norm(_basis_to_world(hpalm_local, rig_fa_basis))
        if rig_fwd is None or rig_acr is None:
            return

        # Solve the full hand basis, then remove only the twist around the
        # forearm axis. This keeps the palm orientation while preventing the
        # wrist mesh from rolling into a visibly distorted shape.
        parent_wq = _q(self._wj(f"FK-Forearm.{side}").getQuat(self.actor))  # type: ignore
        rest_wq = hand_rest.quat * parent_wq
        r0 = _norm(rest_wq.xform(self._hand_basis_in_fk_local[side][0]))
        r1 = _norm(rest_wq.xform(self._hand_basis_in_fk_local[side][1]))
        if r0 is None or r1 is None:
            return

        twist_axis_parent = _norm(self.arm_ctrls[side]["Forearm"].rest_dir_local) or Vec3(0, 0, 1)
        palm_local = self._hand_basis_in_fk_local[side][2]
        new_q = hand_rest.quat
        best_score = -2.0
        for cand_acr in (rig_acr, rig_acr * -1.0):
            delta = _rot_from_basis(r0, r1, rig_fwd, cand_acr)
            full_q = rest_wq * delta * parent_wq.conjugate()
            offset_q = hand_rest.quat.conjugate() * full_q
            swing_q = _remove_twist_from_offset(offset_q, twist_axis_parent)
            cand_q = hand_rest.quat * swing_q
            cand_world_q = cand_q * parent_wq
            cand_palm = _norm(cand_world_q.xform(palm_local))
            score = cand_palm.dot(rig_palm) if cand_palm is not None and rig_palm is not None else -1.0
            if score > best_score:
                best_score = score
                new_q = cand_q

        # Temporal smoothing (EMA with outlier rejection)
        prev = self._prev_hand_quats.get(side)
        if prev is not None:
            # Ensure consistent hemisphere for interpolation
            dot = (prev.getR() * new_q.getR() + prev.getI() * new_q.getI()
                   + prev.getJ() * new_q.getJ() + prev.getK() * new_q.getK())
            if dot < 0.0:
                new_q = Quat(-new_q.getR(), -new_q.getI(),
                             -new_q.getJ(), -new_q.getK())
                dot = -dot
            # Reject outlier frames (>90° jump likely means basis flip)
            angle = math.degrees(2.0 * math.acos(min(dot, 1.0)))
            if angle > 90.0:
                new_q = Quat(prev)
            else:
                a = HAND_QUAT_BLEND_ALPHA
                blended = Quat(
                    prev.getR() * (1.0 - a) + new_q.getR() * a,
                    prev.getI() * (1.0 - a) + new_q.getI() * a,
                    prev.getJ() * (1.0 - a) + new_q.getJ() * a,
                    prev.getK() * (1.0 - a) + new_q.getK() * a,
                )
                blended.normalize()
                new_q = blended

        self._prev_hand_quats[side] = Quat(new_q)
        self._cur_quats[hand_jn] = new_q

    # ------------------------------------------------------------------
    # Finger FK
    # ------------------------------------------------------------------

    def _update_thumb(
        self, side: str, hlms: dict[int, Vec3], cap_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        cur_basis = self._current_hand_basis(side)
        hand_wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor))  # type: ignore

        target_dirs: list[Vec3] = []
        for start_i, end_i in FINGER_LANDMARKS["Thumb"]:
            sp = hlms.get(start_i)
            ep = hlms.get(end_i)
            if sp is None or ep is None:
                return
            cap_dir = _norm(ep - sp)
            if cap_dir is None:
                return
            in_cap = _world_to_basis(cap_dir, cap_basis)
            remap = _norm(_basis_to_world(in_cap, cur_basis))
            if remap is None:
                return
            target_dirs.append(remap)

        parent_wq = hand_wq
        for seg_i, (ctrl, tgt_world) in enumerate(zip(self.finger_ctrls[side]["Thumb"], target_dirs)):
            tgt_par = _norm(parent_wq.conjugate().xform(tgt_world))
            if tgt_par is None:
                self._cur_quats[ctrl.joint_name] = _q(ctrl.rest.quat)
                parent_wq = ctrl.rest.quat * parent_wq
                continue
            rest_fwd = _norm(ctrl.rest.quat.xform(Vec3(1, 0, 0)))
            if rest_fwd is None:
                self._cur_quats[ctrl.joint_name] = _q(ctrl.rest.quat)
                parent_wq = ctrl.rest.quat * parent_wq
                continue

            # The thumb base needs broad freedom, but the distal segments
            # should hinge around the detected curl axis rather than taking
            # an unconstrained shortest arc that can flip backward on one side.
            delta = Quat()
            if seg_i > 0:
                bend_ax = _norm(ctrl.rest.quat.xform(ctrl.curl_axis_local))
                if bend_ax is not None:
                    rest_proj = _norm(rest_fwd - bend_ax * rest_fwd.dot(bend_ax))
                    tgt_proj = _norm(tgt_par - bend_ax * tgt_par.dot(bend_ax))
                    if rest_proj is not None and tgt_proj is not None:
                        angle = math.acos(_clamp(rest_proj.dot(tgt_proj), -1.0, 1.0))
                        if rest_proj.cross(tgt_proj).dot(bend_ax) < 0.0:
                            angle = -angle
                        angle *= THUMB_POSE_STRENGTH * ctrl.curl_sign
                        delta.setFromAxisAngleRad(angle, bend_ax)
                        new_q = ctrl.rest.quat * delta
                        self._cur_quats[ctrl.joint_name] = new_q
                        parent_wq = new_q * parent_wq
                        continue

            cross_ax = _norm(rest_fwd.cross(tgt_par))
            if cross_ax is None:
                self._cur_quats[ctrl.joint_name] = _q(ctrl.rest.quat)
                parent_wq = ctrl.rest.quat * parent_wq
                continue
            angle = math.acos(_clamp(rest_fwd.dot(tgt_par), -1.0, 1.0))
            angle *= THUMB_POSE_STRENGTH
            delta.setFromAxisAngleRad(angle, cross_ax)
            new_q = ctrl.rest.quat * delta
            self._cur_quats[ctrl.joint_name] = new_q
            parent_wq = new_q * parent_wq

    def _update_finger(
        self, side: str, fname: str, hlms: dict[int, Vec3],
        cap_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        """Inter-segment angle FK for non-thumb fingers.

        Measures the bending angle between consecutive segment directions
        rather than the absolute angle in the hand basis.  This correctly
        handles curls past 90° (e.g. a fist) where the previous atan2
        formula would wrap around and produce zero or negative values.
        """
        fcs = self.finger_ctrls[side][fname]
        lm_pairs = FINGER_LANDMARKS[fname]

        # Collect segment directions in capture space
        seg_dirs: list[Vec3] = []
        for si, ei in lm_pairs:
            sp = hlms.get(si)
            ep = hlms.get(ei)
            if sp is None or ep is None:
                return
            d = _norm(ep - sp)
            if d is None:
                return
            seg_dirs.append(d)

        # Use hand forward as the reference direction for the first segment
        prev_dir: Vec3 = cap_basis[0]

        for seg_i, (cur_dir, ctrl) in enumerate(zip(seg_dirs, fcs)):
            # Inter-segment angle (always non-negative)
            dot = _clamp(prev_dir.dot(cur_dir), -1.0, 1.0)
            curl_angle = math.acos(dot)

            # Reduce base-segment influence (natural splay isn't real curl)
            if seg_i == 0:
                curl_angle *= BASE_SEGMENT_CURL_SCALE

            curl_angle = min(curl_angle, MAX_FINGER_CURL_RADIANS)
            curl_angle *= FINGER_POSE_STRENGTH

            offset = Quat()
            offset.setFromAxisAngleRad(curl_angle * ctrl.curl_sign, ctrl.curl_axis_local)
            self._cur_quats[ctrl.joint_name] = ctrl.rest.quat * offset

            prev_dir = cur_dir

    # ------------------------------------------------------------------
    # Pose reset / freeze helpers
    # ------------------------------------------------------------------

    def _reset_pose(self) -> None:
        for jn in self._controlled:
            self._cur_quats[jn] = _q(self._rest[jn].quat)

    def _freeze_joints(self, names) -> None:
        for jn in names:
            r = self._rest[jn]
            self.actor.freezeJoint(
                PART_NAME, jn,
                transform=TransformState.makePosQuatScale(r.pos, self._cur_quats[jn], r.scale),
            )

    def _freeze_all(self) -> None:
        self._freeze_joints(self._controlled)

    # ------------------------------------------------------------------
    # Main update (called by taskMgr each frame)
    # ------------------------------------------------------------------

    def update(self, task):
        if not self.enabled or self.clip is None:
            return task.cont

        fi = self.clip.frame_at_time(task.time, self.fps)

        # On loop restart, clear temporal state
        if self._last_frame is not None and fi < self._last_frame:
            self._prev_hand_bases.clear()
            self._prev_hand_quats.clear()
            self._prev_arm_dirs.clear()
            self._prev_torso = None
            self._cur_torso = None
            self._ref_torso = None
            self._reset_pose()
            self._freeze_all()
            self.actor.update()
        self._last_frame = fi

        pose_lms, hand_lms = self._frame_data(fi)
        self.last_pose_lms = pose_lms
        self.last_hand_lms = hand_lms

        # --- Torso basis (smoothed) ---
        raw_torso = self._capture_torso(pose_lms)
        if raw_torso is not None:
            self._cur_torso = self._smooth_torso(raw_torso)
            # Lock the reference torso from the first valid frame so that
            # noisy per-frame depth rotation does not drag both arms in
            # unison (image-normalised ASLLVD Z is unreliable).
            if self._ref_torso is None:
                self._ref_torso = self._cur_torso

        # --- Arms ---
        for side in SIDES:
            self._update_arms(side, pose_lms)
        self._freeze_joints([
            ctrl.joint_name
            for side in SIDES for ctrl in self.arm_ctrls[side].values()
        ])
        self.actor.update()

        # --- Hands (wrist orientation via forearm-relative remapping) ---
        cap_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        for side in SIDES:
            hlms = hand_lms.get(side)
            if hlms is None:
                self._prev_hand_bases.pop(side, None)
                continue
            raw = self._target_hand_basis(hlms)
            if raw is None:
                self._prev_hand_bases.pop(side, None)
                continue
            cap_bases[side] = self._stabilize_basis(side, raw)

        for side in SIDES:
            hlms = hand_lms.get(side)
            cb = cap_bases.get(side)
            if hlms is None or cb is None:
                continue
            self._update_hand_pose(side, hlms, cb, pose_lms)

        # Freeze hands and update so FK-Hand world quat is correct for fingers
        hand_joints = [f"FK-Hand.{s}" for s in SIDES]
        self._freeze_joints(hand_joints)
        self.actor.update()

        # --- Fingers (after hand freeze, using original capture-space basis) ---
        for side in SIDES:
            hlms = hand_lms.get(side)
            cb = cap_bases.get(side)
            if hlms is None or cb is None:
                continue
            self._update_thumb(side, hlms, cb)
            for fname in ("Index", "Middle", "Ring", "Pinky"):
                self._update_finger(side, fname, hlms, cb)

        finger_joints = [
            ctrl.joint_name
            for side in SIDES
            for chain in self.finger_ctrls[side].values()
            for ctrl in chain
        ]
        self._freeze_joints(finger_joints)
        self.actor.update()
        return task.cont
