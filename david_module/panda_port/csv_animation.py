"""CSV-based FK animator for the *rain* character.

Reads MediaPipe ``pose_world_landmarks`` (hip-centred real-world metres)
and ``hand_landmarks`` (normalised image-space) exported as CSV files from
SignSchool videos (~4 250 signs in ``dataSet/david-dataset/Landmarks/world-pose/``).

Reuses proven vector/basis helpers and rig constants from
``unified_animation`` while providing a fresh animation pipeline.

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
from pathlib import Path
from typing import TYPE_CHECKING

from direct.actor.Actor import Actor
from panda3d.core import Quat, TransformState, Vec3

# ---- shared utilities from the ASLLVD animator ----
from unified_animation import (
    PART_NAME,
    EPSILON,
    SIDES,
    ARM_JOINTS,
    FINGER_LANDMARKS,
    FINGER_JOINTS,
    FINGER_CHILD_JOINTS,
    TORSO_DEPTH_SIGN,
    JointRestTransform,
    ArmControl,
    FingerControl,
    _v,
    _q,
    _norm,
    _clamp,
    _lerp_vec,
    _blend_dir,
    _rot_between,
    _build_basis,
    _world_to_basis,
    _basis_to_world,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants (CSV-specific – may diverge from ASLLVD tuning)
# ---------------------------------------------------------------------------

CSV_DIR = (Path(__file__).resolve().parent / ".." / ".." / "dataSet" / "david-dataset" / "Landmarks" / "world-pose").resolve()
CSV_FILENAME_RE = re.compile(r"^SignSchool\s+(.+?)\s+\[(\d+)x(\d+)\]\.csv$")

DEFAULT_CSV_FPS = 30.0

# MediaPipe pose-world landmark indices
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Temporal smoothing
ARM_BLEND_ALPHA = 0.40
HAND_BASIS_BLEND_ALPHA = 0.30
HAND_QUAT_BLEND_ALPHA = 0.22
TORSO_BLEND_ALPHA = 0.15

# Finger strength
FINGER_POSE_STRENGTH = 0.75
THUMB_POSE_STRENGTH = 0.50
MAX_FINGER_CURL_RADIANS = math.radians(95.0)
BASE_SEGMENT_CURL_SCALE = 0.6


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
    mirrors the patterns proven in ``UnifiedRigAnimator`` while keeping the
    data-flow independent.
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
        self._prev_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._cur_torso: tuple[Vec3, Vec3, Vec3] | None = None
        self._last_frame: int | None = None

        # Build rig data for both sides
        self._rest_torso = self._build_rest_torso()

        for side in SIDES:
            self.arm_ctrls[side] = self._build_arm_ctrls(side)
            self.finger_ctrls[side] = self._build_finger_ctrls(side)
            self._arm_parent_world_q[side] = self._arm_parent_quat(side)
            self._hand_rest_basis[side] = self._build_hand_rest_basis(side)
            hand_wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor))
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
        self._prev_torso = None
        self._cur_torso = None

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
        ls = self._wj("HNG-Upperarm_Parent.L").getPos(self.actor)
        rs = self._wj("HNG-Upperarm_Parent.R").getPos(self.actor)
        lh = self._wj("HNG-Thigh.L").getPos(self.actor)
        rh = self._wj("HNG-Thigh.R").getPos(self.actor)
        sc = (ls + rs) * 0.5
        hc = (lh + rh) * 0.5
        return _build_basis(sc - hc, rs - ls) or (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def _arm_parent_quat(self, side: str) -> Quat:
        name = f"FK-Upperarm.{side}"
        rest = self._rest_of(name)
        wq = _q(self._wj(name).getQuat(self.actor))
        return wq * rest.quat.conjugate()

    def _build_hand_rest_basis(self, side: str) -> tuple[Vec3, Vec3, Vec3]:
        hp = self._wj(f"DEF-Hand.{side}").getPos(self.actor)
        fwd = self._wj(f"DEF-Middle1.{side}").getPos(self.actor) - hp
        across = (
            self._wj(f"DEF-Pinky1.{side}").getPos(self.actor)
            - self._wj(f"DEF-Index1.{side}").getPos(self.actor)
        )
        return _build_basis(fwd, across) or (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    def _build_arm_ctrls(self, side: str) -> dict[str, ArmControl]:
        ctrls: dict[str, ArmControl] = {}
        for seg, (jb, cb) in ARM_JOINTS.items():
            jn, cn = f"{jb}.{side}", f"{cb}.{side}"
            rest = self._rest_of(jn)
            rd = _norm(self._lj(cn).getPos()) or Vec3(0, 0, 1)
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
                    rd = _norm(self._lj(cn).getPos()) or Vec3(0, 0, 1)
                else:
                    parent_name = jnames[i - 1]
                    wj = self._wj(jn)
                    pj = self._wj(parent_name)
                    wdir = _norm(wj.getPos(self.actor) - pj.getPos(self.actor))
                    wq = wj.getQuat(self.actor)
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
        wq = _q(self._wj(joint_name).getQuat(self.actor))
        parent_wq = wq * rest.quat.conjugate()
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
        if self._cur_torso is None:
            return
        cap_torso = self._cur_torso

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
        wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor))
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

        # Capture forearm frame (pose_world only – no coordinate mixing)
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
        cap_fa_basis = _build_basis(cap_fa_dir, cap_ua_dir)
        if cap_fa_basis is None:
            return

        # Hand axes in capture forearm frame
        hfwd_local = _world_to_basis(cap_basis[0], cap_fa_basis)
        hacross_local = _world_to_basis(cap_basis[1], cap_fa_basis)
        hfwd_local = Vec3(hfwd_local.x, hfwd_local.y, hfwd_local.z * TORSO_DEPTH_SIGN)
        hacross_local = Vec3(hacross_local.x, hacross_local.y, hacross_local.z * TORSO_DEPTH_SIGN)

        # Rig forearm frame
        fa_jn = f"FK-Forearm.{side}"
        fa_rest = self._rest_of(fa_jn)
        fa_wq = _q(self._wj(fa_jn).getQuat(self.actor))
        rig_fa_fwd = _norm(fa_wq.xform(fa_rest.quat.conjugate().xform(
            _norm(self._lj(f"FK-Hand.{side}").getPos()) or Vec3(0, 0, 1)
        )))
        ua_jn = f"FK-Upperarm.{side}"
        ua_wq = _q(self._wj(ua_jn).getQuat(self.actor))
        rig_ua_child = _norm(self._lj(fa_jn).getPos()) or Vec3(0, 0, 1)
        rig_ua_dir = _norm(ua_wq.xform(_q(self._rest_of(ua_jn).quat).conjugate().xform(rig_ua_child)))
        if rig_fa_fwd is None or rig_ua_dir is None:
            return
        rig_fa_basis = _build_basis(rig_fa_fwd, rig_ua_dir)
        if rig_fa_basis is None:
            return

        tgt_fwd = _norm(_basis_to_world(hfwd_local, rig_fa_basis))
        tgt_across = _norm(_basis_to_world(hacross_local, rig_fa_basis))
        if tgt_fwd is None or tgt_across is None:
            return
        tgt_basis = _build_basis(tgt_fwd, tgt_across)
        if tgt_basis is None:
            return

        rest_basis = self._hand_rest_basis[side]
        from unified_animation import _rot_from_basis
        delta = _rot_from_basis(rest_basis[0], rest_basis[1], tgt_basis[0], tgt_basis[1])

        parent_wq = _q(self._wj(fa_jn).getQuat(self.actor))
        tgt_world = delta * hand_rest.quat
        new_q = _q(parent_wq.conjugate()) * tgt_world

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

    # ------------------------------------------------------------------
    # Finger FK
    # ------------------------------------------------------------------

    def _update_thumb(
        self, side: str, hlms: dict[int, Vec3],
        cap_basis: tuple[Vec3, Vec3, Vec3],
    ) -> None:
        fcs = self.finger_ctrls[side]["Thumb"]
        lm_pairs = FINGER_LANDMARKS["Thumb"]
        seg_dirs: list[Vec3] = []
        for si, ei in lm_pairs:
            sp, ep = hlms.get(si), hlms.get(ei)
            if sp is None or ep is None:
                return
            d = _norm(ep - sp)
            if d is None:
                return
            seg_dirs.append(d)

        parent_wq = _q(self._wj(f"FK-Hand.{side}").getQuat(self.actor))
        for seg_dir, ctrl in zip(seg_dirs, fcs):
            cap_tgt = _world_to_basis(seg_dir, cap_basis)
            cur_basis = _build_basis(
                _norm(parent_wq.xform(_norm(ctrl.rest.quat.xform(ctrl.rest_dir_local)) or Vec3(1, 0, 0))) or Vec3(1, 0, 0),
                _norm(parent_wq.xform(_norm(ctrl.rest.quat.xform(Vec3(0, 1, 0))) or Vec3(0, 1, 0))) or Vec3(0, 1, 0),
            )
            if cur_basis is None:
                self._cur_quats[ctrl.joint_name] = _q(ctrl.rest.quat)
                parent_wq = parent_wq * ctrl.rest.quat
                continue
            tgt_par = _norm(_basis_to_world(cap_tgt, cur_basis))
            rest_fwd = _norm(ctrl.rest.quat.xform(Vec3(1, 0, 0)))
            if rest_fwd is None or tgt_par is None:
                self._cur_quats[ctrl.joint_name] = _q(ctrl.rest.quat)
                parent_wq = parent_wq * ctrl.rest.quat
                continue
            cross_ax = _norm(rest_fwd.cross(tgt_par))
            if cross_ax is None:
                self._cur_quats[ctrl.joint_name] = _q(ctrl.rest.quat)
                parent_wq = parent_wq * ctrl.rest.quat
                continue
            angle = math.acos(_clamp(rest_fwd.dot(tgt_par), -1.0, 1.0))
            angle *= THUMB_POSE_STRENGTH
            delta = Quat()
            delta.setFromAxisAngleRad(angle, cross_ax)
            new_q = ctrl.rest.quat * delta
            self._cur_quats[ctrl.joint_name] = new_q
            parent_wq = parent_wq * new_q

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
        for seg_i, (cur_dir, ctrl) in enumerate(zip(seg_dirs, fcs)):
            dot = _clamp(prev_dir.dot(cur_dir), -1.0, 1.0)
            curl_angle = math.acos(dot)
            if seg_i == 0:
                curl_angle *= BASE_SEGMENT_CURL_SCALE
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
