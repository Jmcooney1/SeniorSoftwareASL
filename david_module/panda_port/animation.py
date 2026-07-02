"""FK animator for the *rain* character driven by CSV landmark exports.

Reads MediaPipe ``pose_world_landmarks`` (hip-centred real-world metres)
and ``hand_landmarks`` (normalised image-space) exported as CSV files from
SignSchool videos (~4250 signs in ``dataSet/david-dataset/Landmarks/world-pose/``).

Coordinate notes
~~~~~~~~~~~~~~~~
* ``pose_world`` axes after conversion: +X = signer's left, +Y = up, Z ≈ depth.
* ``hand_landmarks`` axes after conversion: approximately the same + Y = up,
  but Z is relative wrist depth, not true world depth.  Directions are
  approximately consistent after normalisation so we re-use the
  forearm-relative hand basis remapping (only directions matter, not scale).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from direct.actor.Actor import Actor
from panda3d.core import LMatrix3f, Quat, TransformState, Vec3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PART_NAME = "modelRoot"
EPSILON = 1.0e-6

# Depth remapping sign -- capture torso Z points backward while rig torso Z
# points forward (toward camera at actor-local -Z).  The flip corrects this
# for arm directions; hand orientation uses forearm-relative remapping instead.
TORSO_DEPTH_SIGN = -1.0

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

# MediaPipe pose-world landmark indices
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# CSV catalogue
def _csv_dir_from_config() -> Path:
    """Read csv_dir from config.json; fall back to dataSet/david-dataset/animations."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "config.json"
    try:
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        rel = cfg.get("csv_dir", "dataSet/david-dataset/animations")
        p = Path(rel)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    except Exception:
        return (repo_root / "dataSet" / "david-dataset" / "animations").resolve()

CSV_DIR = _csv_dir_from_config()
CSV_FILENAME_RE = re.compile(r"^SignSchool\s+(.+?)\s+\[(\d+)x(\d+)\]\.csv$")
DEFAULT_CSV_FPS = 24.0

# Temporal smoothing
ARM_BLEND_ALPHA = 0.40
HAND_BASIS_BLEND_ALPHA = 0.30
HAND_QUAT_BLEND_ALPHA = 0.22
TORSO_BLEND_ALPHA = 0.15

# Finger strength
FINGER_POSE_STRENGTH = 0.90
THUMB_POSE_STRENGTH = 0.50
MAX_FINGER_CURL_RADIANS = math.radians(110.0)


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


def _extract_twist_from_offset(offset_q: Quat, twist_axis_parent: Vec3) -> Quat:
    """Return only the twist component of an offset quaternion around a parent-space axis."""
    axis = _norm(twist_axis_parent)
    if axis is None:
        return Quat.identQuat()

    projected = axis * (
        offset_q.getI() * axis.x
        + offset_q.getJ() * axis.y
        + offset_q.getK() * axis.z
    )
    twist = Quat(offset_q.getR(), projected.x, projected.y, projected.z)
    if twist.lengthSquared() <= EPSILON:
        return Quat.identQuat()
    twist.normalize()
    return twist


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _pw2rig(x: float, y: float, z: float) -> Vec3:
    """``pose_world_landmarks`` → Panda rig space (negate Y for up-positive)."""
    return Vec3(x, -y, z)


def _hl2rig(x: float, y: float, z: float) -> Vec3:
    """``hand_landmarks`` → rig-like space (same axis convention as pose_world)."""
    return Vec3(x, -y, z)


# ---------------------------------------------------------------------------
# CSV catalogue helpers
# ---------------------------------------------------------------------------

def list_csv_signs() -> list[tuple[str, Path]]:
    """Return sorted ``(sign_name, path)`` pairs for every CSV file."""
    if not CSV_DIR.is_dir():
        return []
    results: list[tuple[str, Path]] = []
    for p in CSV_DIR.iterdir():
        m = CSV_FILENAME_RE.match(p.name)
        if m:
            results.append((m.group(1), p))
    results.sort(key=lambda t: t[0].lower())
    return results


def find_csv_sign(name: str) -> Path | None:
    """Find a CSV file for *name* (case-insensitive)."""
    low = name.lower()
    for sign_name, path in list_csv_signs():
        if sign_name.lower() == low:
            return path
    return None


# ---------------------------------------------------------------------------
# CSV sign clip
# ---------------------------------------------------------------------------

class CSVSignClip:
    """Parses a single CSV sign file (pose_world + optional hand landmarks)."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.sign_name = ""
        self.video_width = 0
        self.video_height = 0

        m = CSV_FILENAME_RE.match(self.csv_path.name)
        if m:
            self.sign_name = m.group(1)
            self.video_width = int(m.group(2))
            self.video_height = int(m.group(3))

        self.pose_frames: list[dict[int, Vec3]] = []
        self.left_hand_frames: list[dict[int, Vec3] | None] = []
        self.right_hand_frames: list[dict[int, Vec3] | None] = []
        self.timestamps: list[float] = []

        self._has_left_hand = False
        self._has_right_hand = False
        self._parse()

        self.num_frames: int = len(self.pose_frames)
        self.available: bool = self.num_frames > 0
        self.fps: float = self._compute_fps()

    # ---- parsing ----

    def _parse(self) -> None:
        with self.csv_path.open(newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)

            lh_start = rh_start = -1
            for i, h in enumerate(headers):
                if h == "left_hand_0_x":
                    lh_start = i
                    self._has_left_hand = True
                elif h == "right_hand_0_x":
                    rh_start = i
                    self._has_right_hand = True

            for row in reader:
                if len(row) < 134:
                    continue

                # -- pose_world (33 landmarks) --
                pose: dict[int, Vec3] = {}
                for idx in range(33):
                    base = idx * 4
                    try:
                        x, y, z = float(row[base]), float(row[base + 1]), float(row[base + 2])
                        vis = float(row[base + 3]) if row[base + 3] else 0.0
                        if vis > 0.01:
                            pose[idx] = _pw2rig(x, y, z)
                    except (ValueError, IndexError):
                        pass
                self.pose_frames.append(pose)

                # -- timestamp --
                try:
                    self.timestamps.append(float(row[133]))
                except (ValueError, IndexError):
                    self.timestamps.append(0.0)

                # -- hands --
                self.left_hand_frames.append(
                    self._parse_hand(row, lh_start) if self._has_left_hand and lh_start >= 0 else None
                )
                self.right_hand_frames.append(
                    self._parse_hand(row, rh_start) if self._has_right_hand and rh_start >= 0 else None
                )

    @staticmethod
    def _parse_hand(row: list[str], start: int) -> dict[int, Vec3] | None:
        lms: dict[int, Vec3] = {}
        for idx in range(21):
            base = start + idx * 4
            try:
                xs, ys = row[base], row[base + 1]
                if not xs or not ys:
                    continue
                x, y = float(xs), float(ys)
                zs = row[base + 2]
                z = float(zs) if zs else 0.0
                lms[idx] = _hl2rig(x, y, z)
            except (ValueError, IndexError):
                continue
        return lms if len(lms) >= 10 else None

    def _compute_fps(self) -> float:
        if len(self.timestamps) < 2:
            return DEFAULT_CSV_FPS
        diffs = [
            self.timestamps[i + 1] - self.timestamps[i]
            for i in range(len(self.timestamps) - 1)
            if self.timestamps[i + 1] > self.timestamps[i]
        ]
        if not diffs:
            return DEFAULT_CSV_FPS
        avg_dt = sum(diffs) / len(diffs)
        return (1.0 / avg_dt) if avg_dt > 0 else DEFAULT_CSV_FPS

    # ---- playback ----

    def frame_at_time(self, seconds: float) -> int:
        if not self.available:
            return 0
        total = self.num_frames / self.fps
        if total <= 0:
            return 0
        t = seconds % total
        return min(int(t * self.fps), self.num_frames - 1)

    def frame_data(self, idx: int) -> tuple[dict[int, Vec3], dict[str, dict[int, Vec3]]]:
        pose = self.pose_frames[idx] if idx < len(self.pose_frames) else {}
        lh = self.left_hand_frames[idx] if idx < len(self.left_hand_frames) else None
        rh = self.right_hand_frames[idx] if idx < len(self.right_hand_frames) else None
        hands: dict[str, dict[int, Vec3]] = {}
        if lh is not None:
            hands["L"] = lh
        if rh is not None:
            hands["R"] = rh
        return pose, hands


# ---------------------------------------------------------------------------
# CSV rig animator
# ---------------------------------------------------------------------------

class CSVRigAnimator:
    """FK animator for the *rain* character driven by CSV landmark exports.

    The rig interaction (joint traversal, FK chain, finger curl detection)
    drives the character model from MediaPipe CSV data.
    """

    def __init__(self, actor: Actor, csv_path: Path | None = None) -> None:
        self.actor = actor
        self.clip: CSVSignClip | None = None
        self.enabled = False

        # Public metadata for HUD / UI
        self.selected_sign_name: str | None = None
        self.selected_clip_path: Path | None = None
        self.selected_gloss: str | None = None

        # Exposed for landmark debug visualiser
        self.last_pose_lms: dict[int, Vec3] = {}
        self.last_hand_lms: dict[str, dict[int, Vec3] | None] = {}
        self.hand_world_space: bool = False

        # Joint caches
        self._world_nodes: dict[str, object] = {}
        self._local_nodes: dict[str, object] = {}
        self._rest: dict[str, JointRestTransform] = {}
        self._cur_quats: dict[str, Quat] = {}
        self._controlled: list[str] = []
        self._controlled_set: set[str] = set()

        # Per-side control data
        self.arm_ctrls: dict[str, dict[str, ArmControl]] = {}
        self.finger_ctrls: dict[str, dict[str, tuple[FingerControl, ...]]] = {}
        self._arm_parent_world_q: dict[str, Quat] = {}
        self._hand_rest_basis: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self._hand_basis_in_fk_local: dict[str, tuple[Vec3, Vec3, Vec3]] = {}

        # Temporal smoothing state
        self._prev_arm_dirs: dict[str, dict[str, Vec3]] = {}
        self._prev_hand_bases: dict[str, tuple[Vec3, Vec3, Vec3]] = {}
        self._prev_hand_quats: dict[str, Quat] = {}
        self._prev_forearm_twists: dict[str, Quat] = {}
        self._prev_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._cur_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._ref_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._last_frame: int | None = None

        # Build rig data for both sides
        self._rest_torso = self._build_rest_torso()

        for side in SIDES:
            self.arm_ctrls[side] = self._build_arm_ctrls(side)
            self.finger_ctrls[side] = self._build_finger_ctrls(side)
            self._arm_parent_world_q[side] = self._arm_parent_quat(side)
            self._hand_rest_basis[side] = self._build_hand_rest_basis(side)
            hand_wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor)) # type: ignore
            self._hand_basis_in_fk_local[side] = tuple(
                _norm(hand_wq.conjugate().xform(ax)) or ax
                for ax in self._hand_rest_basis[side]
            )
            for ctrl in self.arm_ctrls[side].values():
                self._register(ctrl.joint_name)
            self._register(f"FK-Hand.{side}")
            for chain in self.finger_ctrls[side].values():
                for fc in chain:
                    self._register(fc.joint_name)

        for jn in self._controlled:
            self._cur_quats[jn] = _q(self._rest[jn].quat)

        if csv_path is not None:
            self.set_clip(CSVSignClip(csv_path))

    # ------------------------------------------------------------------
    # Clip management
    # ------------------------------------------------------------------

    def set_clip(self, clip: CSVSignClip) -> None:
        self.clip = clip
        self.selected_clip_path = clip.csv_path
        self.selected_sign_name = clip.sign_name or clip.csv_path.stem
        self.selected_gloss = clip.sign_name
        self.enabled = clip.available
        self._clear_temporal()
        self._reset_pose()
        self._freeze_all()
        self.actor.update()

    def _clear_temporal(self) -> None:
        self._last_frame = None
        self._prev_arm_dirs.clear()
        self._prev_hand_bases.clear()
        self._prev_hand_quats.clear()
        self._prev_forearm_twists.clear()
        self._prev_torso = None
        self._cur_torso = None
        self._ref_torso = None

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
        n = self._world_nodes.get(name)
        if n is None:
            n = self.actor.exposeJoint(None, PART_NAME, name)
            self._world_nodes[name] = n
        return n

    def _lj(self, name: str):
        n = self._local_nodes.get(name)
        if n is None:
            n = self.actor.exposeJoint(None, PART_NAME, name, localTransform=1)
            self._local_nodes[name] = n
        return n

    def _rest_of(self, name: str) -> JointRestTransform:
        c = self._rest.get(name)
        if c is not None:
            return c
        lj = self._lj(name)
        t = JointRestTransform(pos=_v(lj.getPos()), quat=_q(lj.getQuat()), scale=_v(lj.getScale())) # type: ignore
        self._rest[name] = t
        return t

    # ------------------------------------------------------------------
    # Rig geometry (initialised once from default pose)
    # ------------------------------------------------------------------

    def _build_rest_torso(self) -> tuple[Vec3, Vec3, Vec3]:
        ls = self._wj("HNG-Upperarm_Parent.L").getPos(self.actor) # type: ignore
        rs = self._wj("HNG-Upperarm_Parent.R").getPos(self.actor) # type: ignore
        lh = self._wj("HNG-Thigh.L").getPos(self.actor) # type: ignore
        rh = self._wj("HNG-Thigh.R").getPos(self.actor) # type: ignore
        sc = (ls + rs) * 0.5
        hc = (lh + rh) * 0.5
        return _build_basis(sc - hc, rs - ls) or (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def _arm_parent_quat(self, side: str) -> Quat:
        name = f"FK-Upperarm.{side}"
        rest = self._rest_of(name)
        wq = _q(self._wj(name).getQuat(self.actor)) # type: ignore
        return rest.quat.conjugate() * wq

    def _build_hand_rest_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        hp = self._wj(f"DEF-Hand.{side}").getPos(self.actor) # type: ignore
        fwd = self._wj(f"DEF-Middle1.{side}").getPos(self.actor) - hp # type: ignore
        across = (
            self._wj(f"DEF-Pinky1.{side}").getPos(self.actor) # type: ignore
            - self._wj(f"DEF-Index1.{side}").getPos(self.actor) # type: ignore
        )
        # Left-hand finger-curl direction fix.
        #
        # `fwd × across` is chirality-dependent: on the right hand it yields
        # the back-of-hand normal, but on the left hand it yields the
        # palm-side normal (the mirrored hand geometry inverts the cross
        # product).  `_detect_curl_axis` relies on this vector to decide the
        # sign of positive finger rotation -- specifically `deriv.dot(palm)`
        # determines whether +curl_angle curls fingers inward or outward.
        #
        # Because the rain rig's left-side finger joints have their local
        # axes mirrored (a standard Blender/Rigify convention), the naive
        # unflipped palm vector makes L's positive rotation rotate the
        # fingertips *away* from the palm -- i.e. the fingers bend backward
        # toward the knuckles / back of the hand instead of curling in.
        #
        # Negating `across` on the left side forces the cross product to
        # agree with the right-hand convention (back-of-hand normal), so the
        # sign detection yields matching curl_sign values on both sides.
        # Do NOT remove this without re-deriving curl_sign to handle the
        # mirrored local axes directly.
        if side == "L":
            across = across * -1.0
        return _build_basis(fwd, across) or (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def _build_arm_ctrls(self, side: str) -> dict[str, ArmControl]:
        ctrls: dict[str, ArmControl] = {}
        for seg, (jb, cb) in ARM_JOINTS.items():
            jn, cn = f"{jb}.{side}", f"{cb}.{side}"
            rest = self._rest_of(jn)
            rd = _norm(self._lj(cn).getPos()) or Vec3(0, 0, 1) # type: ignore
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
                    parent_name = jnames[i - 1]
                    wj = self._wj(jn)
                    pj = self._wj(parent_name)
                    wdir = _norm(wj.getPos(self.actor) - pj.getPos(self.actor)) # type: ignore
                    wq = wj.getQuat(self.actor) # type: ignore
                    rd = _norm(wq.conjugate().xform(wdir)) if wdir is not None else Vec3(0, 0, 1)
                    rd = rd or Vec3(0, 0, 1)
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
        hand_basis = self._build_hand_rest_basis(side)
        across_world = hand_basis[1]
        palm_world = hand_basis[2]

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
            deriv = _norm(in_parent.cross(rest_dir_parent))
            if deriv is None:
                continue
            sign = -1.0
            if deriv.dot(palm_parent) > 0.0:
                sign = 1.0
            best_ax, best_align, best_sign = ax, align, sign
        return best_ax, best_sign

    # ------------------------------------------------------------------
    # Torso basis from capture data
    # ------------------------------------------------------------------

    def _capture_torso(self, plms: dict[int, Vec3]) -> tuple[Vec3, Vec3, Vec3] | None:
        ls = plms.get(POSE_LEFT_SHOULDER)
        rs = plms.get(POSE_RIGHT_SHOULDER)
        lh = plms.get(POSE_LEFT_HIP)
        rh = plms.get(POSE_RIGHT_HIP)
        if ls is None or rs is None or lh is None or rh is None:
            return None
        sc = (ls + rs) * 0.5
        hc = (lh + rh) * 0.5
        return _build_basis(sc - hc, rs - ls)

    def _smooth_torso(self, raw: tuple[Vec3, Vec3, Vec3]) -> tuple[Vec3, Vec3, Vec3]:
        prev = self._prev_torso
        if prev is None:
            self._prev_torso = raw
            return raw
        a = TORSO_BLEND_ALPHA
        sx = _norm(_lerp_vec(prev[0], raw[0], a)) or raw[0]
        sy = _norm(_lerp_vec(prev[1], raw[1], a)) or raw[1]
        result = _build_basis(sx, sy) or raw
        self._prev_torso = result
        return result

    def _remap_dir(self, cap_dir: Vec3, cap_torso: tuple[Vec3, Vec3, Vec3]) -> Vec3 | None:
        local = _world_to_basis(cap_dir, cap_torso)
        local = Vec3(local.x, local.y, local.z * TORSO_DEPTH_SIGN)
        return _norm(_basis_to_world(local, self._rest_torso))

    # ------------------------------------------------------------------
    # Arm FK
    # ------------------------------------------------------------------

    def _update_arms(self, side: str, plms: dict[int, Vec3]) -> None:
        if self._ref_torso is None:
            return
        cap_torso = self._ref_torso

        si = POSE_LEFT_SHOULDER if side == "L" else POSE_RIGHT_SHOULDER
        ei = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        wi = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        shoulder, elbow, wrist = plms.get(si), plms.get(ei), plms.get(wi)

        parent_wq = self._arm_parent_world_q[side]

        # Upperarm
        ua = self.arm_ctrls[side]["Upperarm"]
        ua_q = self._cur_quats[ua.joint_name]
        ua_dir = _norm(elbow - shoulder) if shoulder is not None and elbow is not None else None
        if ua_dir is not None:
            remapped = self._remap_dir(ua_dir, cap_torso)
            if remapped is not None:
                prev = self._prev_arm_dirs.get(side, {}).get("Upperarm")
                if prev is not None:
                    remapped = _blend_dir(prev, remapped, ARM_BLEND_ALPHA) or remapped
                tgt = _norm(parent_wq.conjugate().xform(remapped))
                rst = _norm(ua.rest.quat.xform(ua.rest_dir_local))
                if tgt is not None and rst is not None:
                    ua_q = ua.rest.quat * _rot_between(rst, tgt)
                    self._cur_quats[ua.joint_name] = ua_q
                    self._prev_arm_dirs.setdefault(side, {})["Upperarm"] = remapped
        parent_wq = ua_q * parent_wq

        # Forearm
        fa = self.arm_ctrls[side]["Forearm"]
        fa_dir = _norm(wrist - elbow) if elbow is not None and wrist is not None else None
        if fa_dir is not None:
            remapped = self._remap_dir(fa_dir, cap_torso)
            if remapped is not None:
                prev = self._prev_arm_dirs.get(side, {}).get("Forearm")
                if prev is not None:
                    remapped = _blend_dir(prev, remapped, ARM_BLEND_ALPHA) or remapped
                tgt = _norm(parent_wq.conjugate().xform(remapped))
                rst = _norm(fa.rest.quat.xform(fa.rest_dir_local))
                if tgt is not None and rst is not None:
                    self._cur_quats[fa.joint_name] = fa.rest.quat * _rot_between(rst, tgt)
                    self._prev_arm_dirs.setdefault(side, {})["Forearm"] = remapped

    # ------------------------------------------------------------------
    # Hand basis from hand landmarks
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
        fwd = sum(
            [v for v in (palm_c - w, idx_mcp - w, pnk_mcp - w) if v is not None],
            Vec3(0, 0, 0),
        ) * (1.0 / 3.0)
        across_parts = [pnk_mcp - idx_mcp]
        if mid_mcp is not None:
            across_parts.append(mid_mcp - idx_mcp)
        if rng_mcp is not None and mid_mcp is not None:
            across_parts.append(rng_mcp - mid_mcp)
        if rng_mcp is not None:
            across_parts.append(pnk_mcp - rng_mcp)
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
        if prev[2].dot(z) < 0.0:
            y, z = y * -1.0, z * -1.0
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
    # Hand (wrist) FK – forearm-relative remapping
    # ------------------------------------------------------------------

    def _update_hand_pose(
        self, side: str, hlms: dict[int, Vec3],
        cap_basis: tuple[Vec3, Vec3, Vec3],
        plms: dict[int, Vec3],
    ) -> None:
        hand_jn = f"FK-Hand.{side}"
        hand_rest = self._rest_of(hand_jn)

        # Fix 3 requires the torso reference frame for chirality remapping;
        # early-return if it's not ready yet (first frame before pose data
        # has been observed).  The arm chain guards on this separately.
        if self._ref_torso is None:
            return
        cap_torso = self._ref_torso

        # Capture forearm anatomy — raw pose_world directions.
        si = POSE_LEFT_SHOULDER if side == "L" else POSE_RIGHT_SHOULDER
        ei = POSE_LEFT_ELBOW if side == "L" else POSE_RIGHT_ELBOW
        wi = POSE_LEFT_WRIST if side == "L" else POSE_RIGHT_WRIST
        s, e, w = plms.get(si), plms.get(ei), plms.get(wi)
        if s is None or e is None or w is None:
            return
        cap_fa_dir = _norm(w - e)
        cap_ua_dir = _norm(s - e)
        if cap_fa_dir is None or cap_ua_dir is None:
            return

        # Fix 3: torso-consistent chirality correction for hand orientation.
        #
        # Problem history
        # ~~~~~~~~~~~~~~~
        # Capture-space (pose_world after `_pw2rig`) and rig-space differ by
        # a reflection through the torso XY plane — that's what
        # ``TORSO_DEPTH_SIGN`` encodes.  `_remap_dir` applies this reflection
        # correctly for arm direction vectors because it is done in the
        # torso basis, where the Z axis *is* the depth axis by construction.
        #
        # The previous implementation built ``cap_fa_basis`` directly from
        # raw capture coordinates, projected the hand basis vectors into
        # that forearm frame, and then applied ``z *= TORSO_DEPTH_SIGN`` to
        # the forearm-local components.  That flip is only valid if the
        # forearm basis's Z axis corresponds to the torso depth axis — but
        # the forearm basis's Z axis is `forearm_dir × upper_arm_perp`,
        # which rotates arbitrarily with arm configuration.  As the arm
        # moves, the "depth flip" applied to forearm-local coords becomes a
        # flip around a random axis, smearing the chirality correction
        # across the wrist in unpredictable ways.
        #
        # Observed symptoms with the blind forearm-Z flip
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # * "choke" — palm-toward-center orientation reached from the
        #   front, but side view exposed a persistent radial (thumb-side)
        #   deviation baked into the wrist.
        # * "Good, thank you" — skeleton preview showed palm-inward but the
        #   rig rendered palm-toward-screen.  Capture data was correct; the
        #   capture→rig transform mis-rotated the hand.
        # * "binoculars" — palms settled camera-facing rather than
        #   inward-facing, and the motion axis collapsed to ulnar/radial
        #   deviation instead of the required flexion/extension.  This is
        #   exactly what a ~90° error around the forearm axis looks like.
        #
        # Fix 3 strategy
        # ~~~~~~~~~~~~~~
        # Apply the torso-plane reflection to *every* capture-space vector
        # (hand basis + forearm-basis constituents) using `_remap_dir`.
        # After remapping, all inputs live in rig-compatible world space
        # and share a consistent handedness, so the forearm-local
        # components of the hand are meaningful and need no further flip.
        # The resulting hand-to-forearm relative orientation is then
        # re-expressed in the *actual* rig forearm basis (from current FK
        # state) so the hand follows the rig even when the arm FK doesn't
        # perfectly reproduce the captured arm geometry (smoothing, limits).
        rig_hfwd = self._remap_dir(cap_basis[0], cap_torso)
        rig_hacross = self._remap_dir(cap_basis[1], cap_torso)
        rig_fa_dir_from_cap = self._remap_dir(cap_fa_dir, cap_torso)
        rig_ua_dir_from_cap = self._remap_dir(cap_ua_dir, cap_torso)
        if (rig_hfwd is None or rig_hacross is None
                or rig_fa_dir_from_cap is None
                or rig_ua_dir_from_cap is None):
            return
        cap_fa_basis_in_rig = _build_basis(
            rig_fa_dir_from_cap, rig_ua_dir_from_cap
        )
        if cap_fa_basis_in_rig is None:
            return

        # Hand axes in the (rig-compatible) forearm frame.  No axis flip
        # here — the chirality correction has already been applied above to
        # both the hand axes and the forearm-basis constituents, so this is
        # a pure change of basis.
        hfwd_local = _world_to_basis(rig_hfwd, cap_fa_basis_in_rig)
        hacross_local = _world_to_basis(rig_hacross, cap_fa_basis_in_rig)

        # Fix 4: carry the palm normal through the transform EXPLICITLY,
        # because `_remap_dir` is a reflection (det = −1), not a rotation.
        #
        # The bug Fix 3 did not catch
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fix 3 mapped `cap_basis[0]` (fwd) and `cap_basis[1]` (across)
        # through the torso-plane reflection and relied on
        # `_build_basis(tgt_fwd, tgt_across)` to re-derive the palm normal
        # as `tgt_fwd × tgt_across`.  But for any reflection R we have
        #     R(a × b) = −R(a) × R(b)
        # so the cross-product-derived col2 is the NEGATION of the true
        # reflected palm normal.  That sign inversion is what drives:
        #   * "binoculars" — palms end up facing the camera (~180° around
        #     the forearm from palm-inward); motion appears as ulnar/radial
        #     deviation because the swing is resolving the flipped target.
        #   * "Good, thank you" — right palm settles toward-screen despite
        #     the skeleton preview showing palm-inward (the raw capture is
        #     correct; only the capture→rig palm-normal sign is wrong).
        #   * "choke" — smaller visual effect from the front because the
        #     flipped palm is close to an axis of symmetry, but side view
        #     reveals the residual radial deviation.
        #
        # Fix: compute the palm target by explicitly remapping
        # `cap_basis[2]` through the same pipeline (torso reflection →
        # forearm-frame components → rig forearm basis).  The scoring loop
        # then uses this correctly-signed palm as its alignment target, so
        # the `cand_across` sign-search converges to the right palm
        # orientation instead of its 180° mirror.
        #
        # Why we need a separate remap rather than just negating tgt_basis[2]:
        # negating col2 is equivalent here but obscures the reason.  Keeping
        # an explicit remap of cap_basis[2] documents that the palm normal
        # is first-class data from the capture (it carries the true palm
        # direction, independent of any cross-product convention) and must
        # survive the reflection with its sign preserved.
        rig_hpalm = self._remap_dir(cap_basis[2], cap_torso)
        if rig_hpalm is None:
            return
        hpalm_local = _world_to_basis(rig_hpalm, cap_fa_basis_in_rig)

        # Rig forearm frame (from *current* FK pose).  See Fix 3 commentary:
        # using the current rig forearm rather than the capture-derived one
        # keeps the hand glued to the actual rig forearm when arm FK
        # diverges from capture geometry.
        fa_jn = f"FK-Forearm.{side}"
        fa_rest = self._rest_of(fa_jn)
        fa_wq = _q(self._wj(fa_jn).getQuat(self.actor)) # type: ignore
        rig_fa_fwd = _norm(fa_wq.xform(fa_rest.quat.conjugate().xform(
            _norm(self._lj(f"FK-Hand.{side}").getPos()) or Vec3(0, 0, 1) # type: ignore
        )))
        ua_jn = f"FK-Upperarm.{side}"
        ua_wq = _q(self._wj(ua_jn).getQuat(self.actor)) # type: ignore
        rig_ua_child = _norm(self._lj(fa_jn).getPos()) or Vec3(0, 0, 1) # type: ignore
        rig_ua_dir = _norm(ua_wq.xform(_q(self._rest_of(ua_jn).quat).conjugate().xform(rig_ua_child)))
        if rig_fa_fwd is None or rig_ua_dir is None:
            return
        rig_fa_basis = _build_basis(rig_fa_fwd, rig_ua_dir)
        if rig_fa_basis is None:
            return

        tgt_fwd = _norm(_basis_to_world(hfwd_local, rig_fa_basis))
        tgt_across = _norm(_basis_to_world(hacross_local, rig_fa_basis))
        # Fix 4: independently-derived palm target (sign-correct under reflection).
        tgt_palm = _norm(_basis_to_world(hpalm_local, rig_fa_basis))
        if tgt_fwd is None or tgt_across is None or tgt_palm is None:
            return
        tgt_basis = _build_basis(tgt_fwd, tgt_across)
        if tgt_basis is None:
            return

        parent_wq = _q(self._wj(fa_jn).getQuat(self.actor))  # type: ignore
        rest_wq = hand_rest.quat * parent_wq
        r0 = _norm(rest_wq.xform(self._hand_basis_in_fk_local[side][0]))
        r1 = _norm(rest_wq.xform(self._hand_basis_in_fk_local[side][1]))
        if r0 is None or r1 is None:
            return

        # Fix 2: split offset_q into swing (→ hand joint) and twist (→ forearm joint).
        #
        # History:
        #   Fix 0 (original) — stripped twist entirely: palm orientation lost because
        #     the arm chain's _rot_between is zero-twist by construction, so
        #     pronation/supination had nowhere to live.
        #   Fix 1 — full offset_q on the hand: palm orientation now reached the rig,
        #     but the hand joint absorbing the full twist caused wrist-mesh "braiding"
        #     and exaggerated bend during large pronation angles.
        #   Fix 2 (this) — swing on hand, twist on forearm. Equivalent end-effector
        #     world orientation by associativity:
        #         hand_rest * (swing * twist) * fa_local * upperarm_wq
        #       = hand_rest * swing * (twist * fa_local) * upperarm_wq
        #     so pre-multiplying the forearm's local quat by twist_q shifts the twist
        #     up one joint without changing the hand's visible orientation. This spreads
        #     the twist across more of the arm mesh instead of concentrating it at the
        #     wrist. Axis for decomposition is forearm.rest_dir_local because offset_q
        #     is expressed in the forearm's pre-rest local frame.
        twist_axis_parent = _norm(self.arm_ctrls[side]["Forearm"].rest_dir_local) or Vec3(0, 0, 1)
        palm_local = self._hand_basis_in_fk_local[side][2]
        new_q = hand_rest.quat
        new_twist_q: Quat | None = None
        best_score = -2.0
        for cand_across in (tgt_basis[1], tgt_basis[1] * -1.0):
            delta = _rot_from_basis(r0, r1, tgt_basis[0], cand_across)
            full_q = rest_wq * delta * parent_wq.conjugate()
            offset_q = hand_rest.quat.conjugate() * full_q
            swing_q = _remove_twist_from_offset(offset_q, twist_axis_parent)
            twist_q = _extract_twist_from_offset(offset_q, twist_axis_parent)
            # Score with the full offset — final world orientation of the hand is
            # identical whether offset_q sits on the hand alone or is split between
            # hand (swing) and forearm (twist).
            cand_full_q = hand_rest.quat * offset_q
            cand_world_q = cand_full_q * parent_wq
            cand_palm = _norm(cand_world_q.xform(palm_local))
            # Fix 4: score against the explicitly-remapped palm normal
            # (tgt_palm), NOT tgt_basis[2] = tgt_fwd × tgt_across.  Under
            # the torso-plane reflection the cross-product col2 is the
            # negation of the true reflected palm, so using it here sent
            # the sign-search toward the mirror-image palm orientation.
            score = cand_palm.dot(tgt_palm) if cand_palm is not None else -1.0
            if score > best_score:
                best_score = score
                new_q = hand_rest.quat * swing_q
                new_twist_q = twist_q

        # Temporal smoothing with outlier rejection
        prev = self._prev_hand_quats.get(side)
        if prev is not None:
            if abs(prev.dot(new_q)) < math.cos(math.radians(45)):
                return
            new_q = _q(prev + (new_q - prev) * HAND_QUAT_BLEND_ALPHA)
            if new_q.lengthSquared() > EPSILON:
                new_q.normalize()
            else:
                return
        self._prev_hand_quats[side] = _q(new_q)
        self._cur_quats[hand_jn] = new_q

        # Route the twist component to the forearm joint (see fix 2 comment above).
        # Pre-multiplying twist_q onto fa_local applies the twist in the forearm's
        # pre-rest local frame, which is the same frame offset_q was decomposed in.
        # Temporal smoothing: blend the twist against the previous frame's twist to
        # match the hand joint's blending cadence.
        if new_twist_q is not None:
            fa_local = self._cur_quats.get(fa_jn)
            if fa_local is not None:
                prev_twist = self._prev_forearm_twists.get(side)
                if prev_twist is not None:
                    blended = _q(prev_twist + (new_twist_q - prev_twist) * HAND_QUAT_BLEND_ALPHA)
                    if blended.lengthSquared() > EPSILON:
                        blended.normalize()
                        new_twist_q = blended
                self._prev_forearm_twists[side] = _q(new_twist_q)
                self._cur_quats[fa_jn] = _q(new_twist_q * fa_local)

    # ------------------------------------------------------------------
    # Finger FK
    # ------------------------------------------------------------------

    def _update_thumb(
        self, side: str, hlms: dict[int, Vec3],
        cap_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        fcs = self.finger_ctrls[side]["Thumb"]
        lm_pairs = FINGER_LANDMARKS["Thumb"]

        # Collect capture segment directions.
        seg_dirs: list[Vec3] = []
        for si, ei in lm_pairs:
            sp, ep = hlms.get(si), hlms.get(ei)
            if sp is None or ep is None:
                return
            d = _norm(ep - sp)
            if d is None:
                return
            seg_dirs.append(d)

        # --- Base segment: free 3D rotation via hand-basis remap -----------
        cur_basis = self._current_hand_basis(side)
        hand_wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor)) # type: ignore
        in_cap = _world_to_basis(seg_dirs[0], cap_basis)
        tgt_base = _norm(_basis_to_world(in_cap, cur_basis))
        if tgt_base is None:
            return

        ctrl0 = fcs[0]
        tgt_par = _norm(hand_wq.conjugate().xform(tgt_base))
        rest_fwd = _norm(ctrl0.rest.quat.xform(ctrl0.rest_dir_local))
        if tgt_par is None or rest_fwd is None:
            self._cur_quats[ctrl0.joint_name] = _q(ctrl0.rest.quat)
        else:
            cross_ax = _norm(rest_fwd.cross(tgt_par))
            if cross_ax is None:
                self._cur_quats[ctrl0.joint_name] = _q(ctrl0.rest.quat)
            else:
                angle = math.acos(_clamp(rest_fwd.dot(tgt_par), -1.0, 1.0))
                angle *= THUMB_POSE_STRENGTH
                delta = Quat()
                delta.setFromAxisAngleRad(angle, cross_ax)
                self._cur_quats[ctrl0.joint_name] = ctrl0.rest.quat * delta

        # --- Distal segments: inter-segment angle (like _update_finger) ----
        for seg_i in range(1, len(fcs)):
            ctrl = fcs[seg_i]
            prev_dir = seg_dirs[seg_i - 1]
            cur_dir = seg_dirs[seg_i]
            dot = _clamp(prev_dir.dot(cur_dir), -1.0, 1.0)
            curl_angle = math.acos(dot)
            curl_angle = min(curl_angle, MAX_FINGER_CURL_RADIANS)
            curl_angle *= THUMB_POSE_STRENGTH
            offset = Quat()
            offset.setFromAxisAngleRad(curl_angle * ctrl.curl_sign, ctrl.curl_axis_local)
            self._cur_quats[ctrl.joint_name] = ctrl.rest.quat * offset

    def _update_finger(
        self, side: str, fname: str, hlms: dict[int, Vec3],
        cap_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        fcs = self.finger_ctrls[side][fname]
        lm_pairs = FINGER_LANDMARKS[fname]
        seg_dirs: list[Vec3] = []
        for si, ei in lm_pairs:
            sp, ep = hlms.get(si), hlms.get(ei)
            if sp is None or ep is None:
                return
            d = _norm(ep - sp)
            if d is None:
                return
            seg_dirs.append(d)

        prev_dir: Vec3 = cap_basis[0]
        curl_ax_cap = cap_basis[1]  # hand-across ≈ finger curl axis in capture
        for seg_i, (cur_dir, ctrl) in enumerate(zip(seg_dirs, fcs)):
            if seg_i == 0:
                # Project onto the curl plane (perpendicular to the
                # capture curl axis) so splay is excluded and only
                # flexion contributes to the measured angle.
                pp = _norm(prev_dir - curl_ax_cap * prev_dir.dot(curl_ax_cap))
                cp = _norm(cur_dir - curl_ax_cap * cur_dir.dot(curl_ax_cap))
                if pp is not None and cp is not None:
                    curl_angle = math.acos(_clamp(pp.dot(cp), -1.0, 1.0))
                else:
                    curl_angle = 0.0
            else:
                dot = _clamp(prev_dir.dot(cur_dir), -1.0, 1.0)
                curl_angle = math.acos(dot)
            curl_angle = min(curl_angle, MAX_FINGER_CURL_RADIANS)
            curl_angle *= FINGER_POSE_STRENGTH
            offset = Quat()
            offset.setFromAxisAngleRad(curl_angle * ctrl.curl_sign, ctrl.curl_axis_local)
            self._cur_quats[ctrl.joint_name] = ctrl.rest.quat * offset
            prev_dir = cur_dir

    # ------------------------------------------------------------------
    # Pose reset / freeze
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

        fi = self.clip.frame_at_time(task.time)

        if self._last_frame is not None and fi < self._last_frame:
            self._clear_temporal()
            self._reset_pose()
            self._freeze_all()
            self.actor.update()
        self._last_frame = fi

        pose_lms, hand_lms = self.clip.frame_data(fi)
        self.last_pose_lms = pose_lms
        self.last_hand_lms = hand_lms

        # Torso
        raw_torso = self._capture_torso(pose_lms)
        if raw_torso is not None:
            self._cur_torso = self._smooth_torso(raw_torso)
            if self._ref_torso is None:
                self._ref_torso = self._cur_torso

        # Arms
        for side in SIDES:
            self._update_arms(side, pose_lms)
        self._freeze_joints([
            ctrl.joint_name
            for side in SIDES for ctrl in self.arm_ctrls[side].values()
        ])
        self.actor.update()

        # Hands (wrist orientation)
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

        hand_joints = [f"FK-Hand.{s}" for s in SIDES]
        self._freeze_joints(hand_joints)
        self.actor.update()

        # Fingers
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
