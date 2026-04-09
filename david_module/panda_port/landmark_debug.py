"""Landmark debug overlay for ASL sign animation.

Toggle with **V**:  OFF → 3D (always-on-top) → 2D (flat overlay) → OFF

3D mode
    Landmarks and bones rendered on top of the character in rig world-space
    so you can orbit freely and inspect the skeleton from any angle.
    Arm segment lengths are matched to the model's actual bone lengths
    while preserving the capture data's joint directions.

2D mode
    A flat MediaPipe-style skeleton drawn on ``aspect2d``.  The camera
    locks to a front view and zooms out slightly so the overlay aligns
    with the model beneath it.
"""

from __future__ import annotations

import math

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from panda3d.core import LineSegs, NodePath, Point2, Point3, TextNode, Vec3, Vec4

from animation import (
    TORSO_DEPTH_SIGN,
    _v,
    _norm,
    _build_basis,
    _world_to_basis,
    _basis_to_world,
)

# -----------------------------------------------------------------------
# Feature switches — set to True to render face / leg landmarks
# -----------------------------------------------------------------------

RENDER_FACE = False
RENDER_LEGS = False

# -----------------------------------------------------------------------
# Skeleton topology  (correct MediaPipe connections)
# -----------------------------------------------------------------------

# Indices of landmarks that belong to face or legs, used to filter markers.
_FACE_INDICES = frozenset(range(0, 11))        # 0-10
_LEG_INDICES = frozenset(range(25, 33))         # 25-32

_POSE_CONNECTIONS_FACE = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]

_POSE_CONNECTIONS_TORSO_ARMS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
]

_POSE_CONNECTIONS_LEGS = [
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _active_pose_connections() -> list[tuple[int, int]]:
    """Return the currently enabled pose connections (respects feature flags)."""
    out = list(_POSE_CONNECTIONS_TORSO_ARMS)
    if RENDER_FACE:
        out.extend(_POSE_CONNECTIONS_FACE)
    if RENDER_LEGS:
        out.extend(_POSE_CONNECTIONS_LEGS)
    return out


def _active_pose_indices() -> frozenset[int]:
    """Return the set of landmark indices to skip based on feature flags."""
    skip: set[int] = set()
    if not RENDER_FACE:
        skip |= _FACE_INDICES
    if not RENDER_LEGS:
        skip |= _LEG_INDICES
    return frozenset(skip)


# -----------------------------------------------------------------------
# Colours
# -----------------------------------------------------------------------

POSE_BONE_COLOR = Vec4(0.2, 0.75, 1.0, 1.0)
POSE_MARKER_COLOR = Vec4(1.0, 1.0, 1.0, 1.0)
LEFT_HAND_COLOR = Vec4(0.15, 1.0, 0.35, 1.0)
RIGHT_HAND_COLOR = Vec4(1.0, 0.5, 0.15, 1.0)

# -----------------------------------------------------------------------
# Mode constants
# -----------------------------------------------------------------------

MODE_OFF = 0
MODE_3D = 1
MODE_2D = 2
_MODE_COUNT = 3
_MODE_LABELS = ("OFF", "3D Overlay", "2D Overlay")

TOGGLE_KEY = "v"

PART_NAME = "modelRoot"


# -----------------------------------------------------------------------
# Visualiser
# -----------------------------------------------------------------------

class LandmarkVisualizer(DirectObject):
    """Draw pose + hand landmarks over the Panda3D scene.

    Parameters
    ----------
    world : ShowBase
        The running Panda3D application.
    actor : Actor
        The character model.
    camera_controller : CameraController | None
        Optional orbit-camera controller; paused in 2D mode.
    hand_world_space : bool
        ``True`` when hand landmarks share the same coordinate system as
        pose landmarks (ASLLVD).  ``False`` when hands are in image-normalised
        space (CSV) and must be re-anchored at the wrist.
    """

    def __init__(
        self,
        world,
        actor,
        camera_controller=None,
        hand_world_space: bool = True,
    ) -> None:
        super().__init__()
        self.world = world
        self.actor = actor
        self.camera_controller = camera_controller
        self.hand_world_space = hand_world_space
        self.mode: int = MODE_OFF

        # ---- 3D overlay (parented to actor → actor-local coords) ----
        self._3d_root: NodePath = actor.attachNewNode("landmark-3d")
        self._3d_root.setDepthTest(False)
        self._3d_root.setDepthWrite(False)
        self._3d_root.setBin("fixed", 100)
        self._3d_root.setLightOff()
        self._3d_root.hide()
        self._3d_geom: NodePath | None = None

        # ---- 2D overlay (lives on aspect2d) ----
        a2d = getattr(world, "aspect2d", None)
        if a2d is not None:
            self._2d_root: NodePath = a2d.attachNewNode("landmark-2d")
        else:
            self._2d_root = NodePath("landmark-2d-orphan")
        self._2d_root.hide()
        self._2d_geom: NodePath | None = None

        # ---- rig reference geometry (actor-local, computed once) ----
        self._rig_ls = self._jpos("HNG-Upperarm_Parent.L")
        self._rig_rs = self._jpos("HNG-Upperarm_Parent.R")
        self._rig_lh = self._jpos("HNG-Thigh.L")
        self._rig_rh = self._jpos("HNG-Thigh.R")
        self._rig_sc = (self._rig_ls + self._rig_rs) * 0.5
        self._rig_hc = (self._rig_lh + self._rig_rh) * 0.5
        self._rig_torso = _build_basis(
            self._rig_sc - self._rig_hc, self._rig_rs - self._rig_ls,
        ) or (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
        self._rig_shoulder_dist = (self._rig_rs - self._rig_ls).length()

        # Reference capture torso (set once from first 3D frame, not per-frame)
        self._ref_cap_torso: tuple[Vec3, Vec3, Vec3] | None = None

        # Rig bone lengths (rest pose: shoulder→elbow, elbow→wrist)
        self._rig_bones: dict[str, dict[str, float]] = {}
        for side in ("L", "R"):
            sj = self._jpos(f"HNG-Upperarm_Parent.{side}")
            ej = self._jpos(f"FK-Forearm.{side}")
            wj = self._jpos(f"FK-Hand.{side}")
            self._rig_bones[side] = {
                "upper": (ej - sj).length(),
                "lower": (wj - ej).length(),
            }

        # Rig hand size – full hand (wrist → middle fingertip)
        # Used by CSV _anchor_hand_3d for stable scaling.
        self._rig_hand_size: dict[str, float] = {}
        # Rig metacarpal size (wrist → middle MCP)
        # Used by ASLLVD _anchor_hand_at_wrist_world where landmarks
        # are already in world-space and only need MCP-level rescaling.
        self._rig_mcp_size: dict[str, float] = {}
        for side in ("L", "R"):
            try:
                wp = self._jpos(f"DEF-Hand.{side}")
                mcp = self._jpos(f"DEF-Middle1.{side}")
                tip = self._jpos(f"DEF-Middle3.{side}")
                self._rig_hand_size[side] = (tip - wp).length()
                self._rig_mcp_size[side] = (mcp - wp).length()
            except Exception:
                self._rig_hand_size[side] = 0.12
                self._rig_mcp_size[side] = 0.025

        # ---- camera save-state for 2D lock ----
        self._saved_cam_pos: Vec3 | None = None
        self._saved_cam_hpr: Vec3 | None = None
        self._cam_task_paused: bool = False

        # ---- HUD label (always visible) ----
        self._mode_text: OnscreenText | None = None
        self._refresh_label()

        # ---- key binding ----
        self.accept(TOGGLE_KEY, self.toggle_mode)

    # ----- helpers --------------------------------------------------

    def _jpos(self, name: str) -> Vec3:
        return _v(self.actor.exposeJoint(None, PART_NAME, name).getPos(self.actor))

    # ----- mode cycling ----------------------------------------------

    def toggle_mode(self) -> None:
        old = self.mode
        self.mode = (self.mode + 1) % _MODE_COUNT

        # leave old mode
        if old == MODE_3D:
            self._3d_root.hide()
            self._clear_node("_3d_geom")
        elif old == MODE_2D:
            self._2d_root.hide()
            self._clear_node("_2d_geom")
            self._exit_2d_camera()

        # enter new mode
        if self.mode == MODE_3D:
            self._3d_root.show()
            self._ref_cap_torso = None  # re-calibrate from next frame
        elif self.mode == MODE_2D:
            self._2d_root.show()
            self._enter_2d_camera()

        self._refresh_label()

    # ----- HUD label (always visible) --------------------------------

    def _refresh_label(self) -> None:
        if self._mode_text is not None:
            self._mode_text.destroy()
            self._mode_text = None
        try:
            aspect = float(self.world.getAspectRatio())
        except Exception:
            aspect = 1.0
        lines = [
            f"Skeleton: {_MODE_LABELS[self.mode]}  [V] cycle",
            "[F] toggle eye-tracking",
        ]
        self._mode_text = OnscreenText(
            text="\n".join(lines),
            parent=getattr(self.world, "aspect2d", None) or self._2d_root,
            pos=(aspect - 0.08, 0.92),
            scale=0.038,
            align=TextNode.ARight,
            fg=(1.0, 1.0, 0.3, 0.9),
            bg=(0.08, 0.1, 0.14, 0.65),
            mayChange=False,
        )

    # ----- 2D camera lock / unlock -----------------------------------

    def _enter_2d_camera(self) -> None:
        cam = getattr(self.world, "camera", None) or getattr(self.world, "cam", None)
        if cam is None:
            return
        self._saved_cam_pos = _v(cam.getPos())
        self._saved_cam_hpr = _v(cam.getHpr())

        # Pause the orbit controller so it doesn't fight us
        if self.camera_controller is not None:
            self._cam_task_paused = True
            try:
                self.world.taskMgr.remove("orbit-camera-controller")
            except Exception:
                pass

        # Front view, slightly zoomed out
        from panda_core import (
            CAMERA_TARGET_X,
            CAMERA_TARGET_Y,
            CAMERA_TARGET_Z,
            camera_target_point,
        )
        cam.setPos(CAMERA_TARGET_X, CAMERA_TARGET_Y - 2.2, CAMERA_TARGET_Z)
        cam.lookAt(camera_target_point())

    def _exit_2d_camera(self) -> None:
        cam = getattr(self.world, "camera", None) or getattr(self.world, "cam", None)
        if cam is None:
            return
        if self._saved_cam_pos is not None:
            cam.setPos(self._saved_cam_pos)
        if self._saved_cam_hpr is not None:
            cam.setHpr(self._saved_cam_hpr)

        # Resume orbit controller
        if self._cam_task_paused and self.camera_controller is not None:
            try:
                self.world.taskMgr.add(
                    self.camera_controller._update,
                    "orbit-camera-controller",
                )
            except Exception:
                pass
            self._cam_task_paused = False

    # ----- per-frame entry point -------------------------------------

    def update(self, pose_lms: dict, hand_lms: dict) -> None:
        """Call once per frame with the animator's current landmarks."""
        if self.mode == MODE_OFF:
            return
        if self.mode == MODE_3D:
            self._update_3d(pose_lms, hand_lms)
        else:
            self._update_2d(pose_lms, hand_lms)

    # =================================================================
    #  3D overlay — bone-length-matched skeleton
    # =================================================================

    def _clear_node(self, attr: str) -> None:
        node = getattr(self, attr, None)
        if node is not None:
            node.removeNode()
        setattr(self, attr, None)

    def _capture_torso_basis(self, pose_lms):
        """Return (cap_sc, cap_torso_basis, cap_shoulder_dist) or None."""
        ls = pose_lms.get(11)
        rs = pose_lms.get(12)
        lh = pose_lms.get(23)
        rh = pose_lms.get(24)
        if ls is None or rs is None:
            return None
        cap_sc = (ls + rs) * 0.5
        # Use hips if available, otherwise fall back to a default down vector
        if lh is not None and rh is not None:
            cap_hc = (lh + rh) * 0.5
            primary = cap_sc - cap_hc
        else:
            primary = Vec3(0, 1, 0)  # assume upright
        cap_torso = _build_basis(primary, rs - ls)
        if cap_torso is None:
            return None
        cap_sd = (rs - ls).length()
        if cap_sd < 1e-6:
            return None
        return cap_sc, cap_torso, cap_sd

    def _to_rig(self, pos, cap_sc, cap_torso, scale):
        """Map a capture-space position → actor-local rig position."""
        offset = pos - cap_sc
        local = _world_to_basis(offset, cap_torso)
        local = Vec3(local.x, local.y, local.z * TORSO_DEPTH_SIGN)
        return self._rig_sc + _basis_to_world(local, self._rig_torso) * scale

    def _direction_preserving_chain(self, cap_shoulder, cap_elbow, cap_wrist, side, cap_sc, cap_torso, scale):
        """Place shoulder/elbow/wrist using capture directions but rig bone lengths."""
        rig_upper_len = self._rig_bones[side]["upper"]
        rig_lower_len = self._rig_bones[side]["lower"]

        rig_shoulder = self._to_rig(cap_shoulder, cap_sc, cap_torso, scale)

        # Upper arm: capture direction, rig length
        cap_ua_dir = cap_elbow - cap_shoulder
        ua_len = cap_ua_dir.length()
        if ua_len > 1e-6:
            ua_dir_rig = self._to_rig(cap_elbow, cap_sc, cap_torso, scale) - rig_shoulder
            ua_dir_len = ua_dir_rig.length()
            if ua_dir_len > 1e-6:
                ua_dir_rig = ua_dir_rig * (1.0 / ua_dir_len)
            else:
                ua_dir_rig = Vec3(0, 0, -1)
        else:
            ua_dir_rig = Vec3(0, 0, -1)
        rig_elbow = rig_shoulder + ua_dir_rig * rig_upper_len

        # Forearm: capture direction, rig length
        cap_fa_dir = cap_wrist - cap_elbow
        fa_len = cap_fa_dir.length()
        if fa_len > 1e-6:
            fa_dir_rig = self._to_rig(cap_wrist, cap_sc, cap_torso, scale) - self._to_rig(cap_elbow, cap_sc, cap_torso, scale)
            fa_dir_len = fa_dir_rig.length()
            if fa_dir_len > 1e-6:
                fa_dir_rig = fa_dir_rig * (1.0 / fa_dir_len)
            else:
                fa_dir_rig = Vec3(0, 0, -1)
        else:
            fa_dir_rig = Vec3(0, 0, -1)
        rig_wrist = rig_elbow + fa_dir_rig * rig_lower_len

        return rig_shoulder, rig_elbow, rig_wrist

    def _anchor_hand_3d(self, hlms, rig_wrist, side, rig_pose):
        """Scale, orient, and translate CSV hand landmarks so wrist(0)
        sits at *rig_wrist*, the hand is rotated to match the pose
        wrist-triangle orientation, and overall hand size matches the
        rig model's hand."""
        w0 = hlms.get(0)
        if w0 is None:
            return {}

        # Pose wrist-triangle landmarks (already in rig space)
        wi, ii, pi = (15, 19, 17) if side == "L" else (16, 20, 18)
        p_w = rig_pose.get(wi)
        p_idx = rig_pose.get(ii)
        p_pnk = rig_pose.get(pi)

        # --- scale factor: rig full-hand vs capture palm size ---
        # MCP joints (5, 9, 13, 17) are structurally rigid relative to
        # the wrist, so their average distance from wrist(0) is *stable*
        # across all hand poses — unlike fingertips which move enormously
        # between an open hand and a fist.
        #
        # We scale so that the capture MCP region maps to the rig's full
        # hand length (DEF-Hand → DEF-Middle3), corrected by a fixed
        # anatomical proportion (MCP ≈ 53% of full hand in MediaPipe).
        mcp_lms = [hlms.get(i) for i in (5, 9, 13, 17)]
        mcp_dists = [(lm - w0).length() for lm in mcp_lms if lm is not None]
        cap_mcp = sum(mcp_dists) / len(mcp_dists) if mcp_dists else 0.10
        rig_hs = self._rig_hand_size.get(side, 0.12)
        # MCP avg is ~53% of full hand span in MediaPipe image-space
        s = rig_hs * 0.53 / max(cap_mcp, 1e-6)

        # --- source basis from hand landmark palm plane ---
        h_idx_mcp = hlms.get(5)    # index MCP
        h_pnk_mcp = hlms.get(17)   # pinky MCP
        src_basis = None
        if h_idx_mcp is not None and h_pnk_mcp is not None:
            h_primary = (h_idx_mcp + h_pnk_mcp) * 0.5 - w0  # wrist → knuckle centre
            h_secondary = h_idx_mcp - h_pnk_mcp              # across palm
            src_basis = _build_basis(h_primary, h_secondary)

        # --- target basis from pose wrist triangle ---
        tgt_basis = None
        if p_w is not None and p_idx is not None and p_pnk is not None:
            p_primary = (p_idx + p_pnk) * 0.5 - p_w  # wrist → finger centre
            p_secondary = p_idx - p_pnk               # across palm
            tgt_basis = _build_basis(p_primary, p_secondary)

        # --- rotate, scale, translate ---
        if src_basis is not None and tgt_basis is not None:
            # The target basis is built from rig_pose landmarks that went
            # through _to_rig, which applies TORSO_DEPTH_SIGN (−1) to the
            # depth component.  This flips the wrist-triangle's palm normal
            # (Z-axis of the basis).  Negate the decomposed Z so the palm
            # faces the correct direction after recomposition.
            def _depth_corrected(v: Vec3) -> Vec3:
                c = _world_to_basis(v, src_basis)
                return Vec3(c.x, c.y, -c.z)

            return {
                i: rig_wrist + _basis_to_world(
                    _depth_corrected(p - w0), tgt_basis
                ) * s
                for i, p in hlms.items()
            }

        # Fallback: scale + translate only (no orientation data)
        return {i: rig_wrist + (p - w0) * s for i, p in hlms.items()}

    def _update_3d(self, pose_lms, hand_lms):
        self._clear_node("_3d_geom")
        tb = self._capture_torso_basis(pose_lms)
        if tb is None:
            return
        cap_sc, cap_torso, cap_sd = tb
        # Lock the reference torso from the first valid frame to avoid
        # noisy monocular depth rotating both arms in unison.
        if self._ref_cap_torso is None:
            self._ref_cap_torso = cap_torso
        cap_torso = self._ref_cap_torso
        scale = self._rig_shoulder_dist / cap_sd

        ls = LineSegs("lm-debug-3d")
        skip = _active_pose_indices()

        # --- Build rig-space pose landmarks ---
        # Start with simple affine mapping for torso + non-arm landmarks
        rig_pose: dict[int, Vec3] = {}
        for i, p in pose_lms.items():
            if i in skip:
                continue
            rig_pose[i] = self._to_rig(p, cap_sc, cap_torso, scale)

        # Override arm chain landmarks with bone-length–matched positions
        arm_map = {
            "L": (11, 13, 15),  # shoulder, elbow, wrist
            "R": (12, 14, 16),
        }
        rig_wrists: dict[str, Vec3] = {}
        for side, (si, ei, wi) in arm_map.items():
            cs = pose_lms.get(si)
            ce = pose_lms.get(ei)
            cw = pose_lms.get(wi)
            if cs is not None and ce is not None and cw is not None:
                rs, re, rw = self._direction_preserving_chain(
                    cs, ce, cw, side, cap_sc, cap_torso, scale,
                )
                rig_pose[si] = rs
                rig_pose[ei] = re
                rig_pose[wi] = rw
                rig_wrists[side] = rw

                # Also remap the wrist sub-landmarks (thumb/index/pinky tips
                # attached to wrist in the pose connections: 17-22)
                wrist_children = {
                    "L": [17, 19, 21],
                    "R": [18, 20, 22],
                }
                for ci in wrist_children[side]:
                    cp = pose_lms.get(ci)
                    if cp is not None:
                        # direction from capture wrist → child, scaled by
                        # rig forearm length / capture forearm length
                        cap_fa_len = (cw - ce).length()
                        rig_fa_len = self._rig_bones[side]["lower"]
                        child_scale = rig_fa_len / max(cap_fa_len, 1e-6)
                        child_dir = self._to_rig(cp, cap_sc, cap_torso, scale) - self._to_rig(cw, cap_sc, cap_torso, scale)
                        rig_pose[ci] = rw + child_dir * (child_scale / max(scale, 1e-6))

        pose_conns = _active_pose_connections()
        _draw_bones(ls, rig_pose, pose_conns, POSE_BONE_COLOR, 2.5)
        _draw_markers_3d(ls, rig_pose, POSE_MARKER_COLOR, 5.0, 0.008)

        # --- hands ---
        for side, hlms in hand_lms.items():
            if hlms is None:
                continue
            color = LEFT_HAND_COLOR if side == "L" else RIGHT_HAND_COLOR

            wrist_pos = rig_wrists.get(side)
            if wrist_pos is None:
                wi = 15 if side == "L" else 16
                wrist_pos = rig_pose.get(wi)
            if wrist_pos is None:
                continue

            if self.hand_world_space:
                # ASLLVD: hand landmarks in same world space as pose
                # Map wrist(0) → rig wrist, scale hand to rig hand size
                rig_hand = self._anchor_hand_at_wrist_world(
                    hlms, wrist_pos, side, cap_sc, cap_torso, scale,
                )
            else:
                # CSV: hand landmarks in image-normalised space
                rig_hand = self._anchor_hand_3d(hlms, wrist_pos, side, rig_pose)

            _draw_bones(ls, rig_hand, HAND_CONNECTIONS, color, 1.5)
            _draw_markers_3d(ls, rig_hand, color, 3.5, 0.005)

        node = ls.create()
        self._3d_geom = self._3d_root.attachNewNode(node)

    def _anchor_hand_at_wrist_world(self, hlms, rig_wrist, side, cap_sc, cap_torso, scale):
        """For ASLLVD (hand_world_space=True): map hand landmarks into rig space
        with the wrist pinned at *rig_wrist* and hand scaled to rig hand size."""
        w0 = hlms.get(0)
        if w0 is None:
            return {}
        # Map all hand landmarks through the torso affine, then re-anchor
        mapped = {i: self._to_rig(p, cap_sc, cap_torso, scale) for i, p in hlms.items()}
        mapped_w0 = mapped.get(0)
        if mapped_w0 is None:
            return mapped
        # Use average of 4 MCP landmarks for stable size reference
        # (same approach as CSV _anchor_hand_3d).
        mcp_dists = [
            (mapped[i] - mapped_w0).length()
            for i in (5, 9, 13, 17) if i in mapped
        ]
        cap_mcp = sum(mcp_dists) / len(mcp_dists) if mcp_dists else 0.06
        rig_hs = self._rig_hand_size.get(side, 0.12)
        # MCP avg ≈ 53% of full hand span — same proportion as CSV path
        hs = rig_hs * 0.53 / max(cap_mcp, 1e-4)
        return {i: rig_wrist + (p - mapped_w0) * hs for i, p in mapped.items()}

    # =================================================================
    #  2D overlay — model-proportioned flat skeleton
    # =================================================================

    def _update_2d(self, pose_lms, hand_lms):
        self._clear_node("_2d_geom")
        if not pose_lms:
            return

        skip = _active_pose_indices()

        # Use only torso landmarks (shoulders + hips) to compute centering
        # so that face/leg data doesn't pull the skeleton off-centre.
        torso_ids = [11, 12, 23, 24]
        anchor_pts = [pose_lms[i] for i in torso_ids if i in pose_lms]
        if len(anchor_pts) < 2:
            # Fall back to all available landmarks
            anchor_pts = [p for i, p in pose_lms.items() if i not in skip]
        if not anchor_pts:
            return

        xs = [p.x for p in anchor_pts]
        ys = [p.y for p in anchor_pts]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5

        # Scale so that capture shoulder width == rig shoulder width in
        # aspect2d units.  The 2D camera sits at Y = -2.2; at that distance
        # (and default FOV 50) 1 unit actor-local ≈ 0.43 aspect2d units.
        # We calibrate from shoulder distance so the 2D skeleton sits on
        # top of the model.
        cap_ls = pose_lms.get(11)
        cap_rs = pose_lms.get(12)
        if cap_ls is not None and cap_rs is not None:
            cap_sd = (cap_rs - cap_ls).length()
        else:
            cap_sd = max(max(xs) - min(xs), 0.01)

        # Target shoulder width on aspect2d  (empirically tuned)
        target_shoulder_a2d = 0.30
        s = target_shoulder_a2d / max(cap_sd, 1e-6)

        # Vertical offset: project the rig's torso centre onto aspect2d
        # so the skeleton aligns with the model beneath it.
        oy = 0.0
        try:
            rig_tc = Point3((self._rig_sc + self._rig_hc) * 0.5)
            cam = self.world.cam
            rig_tc_cam = cam.getRelativePoint(self.actor, rig_tc)
            proj = Point2()
            if cam.node().getLens().project(rig_tc_cam, proj):
                oy = proj.y
        except Exception:
            pass

        def _to2d(p: Vec3) -> Vec3:
            return Vec3((p.x - cx) * s, 0, (p.y - cy) * s + oy)

        ls_seg = LineSegs("lm-debug-2d")

        # --- pose ---
        sp = {i: _to2d(p) for i, p in pose_lms.items() if i not in skip}
        pose_conns = _active_pose_connections()
        _draw_bones(ls_seg, sp, pose_conns, POSE_BONE_COLOR, 2.5)
        _draw_markers_2d(ls_seg, sp, POSE_MARKER_COLOR, 6.0, 0.018)

        # --- hands ---
        for side, hlms in hand_lms.items():
            if hlms is None:
                continue
            color = LEFT_HAND_COLOR if side == "L" else RIGHT_HAND_COLOR

            # CSV: anchor at pose wrist screen position, scale to match
            wi = 15 if side == "L" else 16
            pw = pose_lms.get(wi)
            w0 = hlms.get(0)
            if pw is None or w0 is None:
                continue
            sw = _to2d(pw)
            mid = hlms.get(9)
            cap_hs = (mid - w0).length() if mid is not None else 0.15
            # Target hand screen-size: proportion of shoulder width
            hs = (target_shoulder_a2d * 0.37 / max(cap_hs, 1e-4))
            sh = {
                i: Vec3(sw.x + (p.x - w0.x) * hs, 0, sw.z + (p.y - w0.y) * hs)
                for i, p in hlms.items()
            }

            _draw_bones(ls_seg, sh, HAND_CONNECTIONS, color, 1.5)
            _draw_markers_2d(ls_seg, sh, color, 4.0, 0.010)

        node = ls_seg.create()
        self._2d_geom = self._2d_root.attachNewNode(node)

    # ----- cleanup ---------------------------------------------------

    def destroy(self) -> None:
        self.ignoreAll()
        self._clear_node("_3d_geom")
        self._clear_node("_2d_geom")
        if self._mode_text is not None:
            self._mode_text.destroy()
        self._3d_root.removeNode()
        self._2d_root.removeNode()
        if self._cam_task_paused:
            self._exit_2d_camera()


# -----------------------------------------------------------------------
# Drawing primitives (module-level, stateless)
# -----------------------------------------------------------------------

def _draw_bones(
    ls: LineSegs,
    lms: dict[int, Vec3],
    connections: list[tuple[int, int]],
    color: Vec4,
    thickness: float,
) -> None:
    ls.setThickness(thickness)
    ls.setColor(color)
    for a, b in connections:
        pa = lms.get(a)
        pb = lms.get(b)
        if pa is not None and pb is not None:
            ls.moveTo(pa)
            ls.drawTo(pb)


def _draw_markers_3d(
    ls: LineSegs,
    lms: dict[int, Vec3],
    color: Vec4,
    thickness: float,
    size: float,
) -> None:
    """Small 3-axis crosses in 3D space."""
    ls.setThickness(thickness)
    ls.setColor(color)
    for pos in lms.values():
        ls.moveTo(pos + Vec3(size, 0, 0))
        ls.drawTo(pos - Vec3(size, 0, 0))
        ls.moveTo(pos + Vec3(0, size, 0))
        ls.drawTo(pos - Vec3(0, size, 0))
        ls.moveTo(pos + Vec3(0, 0, size))
        ls.drawTo(pos - Vec3(0, 0, size))


def _draw_markers_2d(
    ls: LineSegs,
    lms: dict[int, Vec3],
    color: Vec4,
    thickness: float,
    size: float,
) -> None:
    """Small 2-axis crosses on the XZ screen plane."""
    ls.setThickness(thickness)
    ls.setColor(color)
    for pos in lms.values():
        ls.moveTo(pos + Vec3(size, 0, 0))
        ls.drawTo(pos - Vec3(size, 0, 0))
        ls.moveTo(pos + Vec3(0, 0, size))
        ls.drawTo(pos - Vec3(0, 0, size))
