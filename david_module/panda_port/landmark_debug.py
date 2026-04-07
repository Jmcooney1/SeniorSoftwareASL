"""Landmark debug overlay for ASL sign animation.

Toggle with **V**:  OFF → 3D (always-on-top) → 2D (flat overlay) → OFF

3D mode
    Landmarks and bones rendered on top of the character in rig world-space.
    Bone lengths are matched to the model's actual rig proportions so
    shoulder, elbow, and wrist landmarks line up with the character's joints.
    Viewable from any angle.

2D mode
    A flat MediaPipe-style skeleton drawn on ``aspect2d``.  The camera
    locks to a front view.  The skeleton is scaled so its shoulder width
    matches the model's on-screen shoulder width and is centred over the
    character's torso.
"""

from __future__ import annotations

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from panda3d.core import LineSegs, LPoint2f, LPoint3f, NodePath, TextNode, Vec3, Vec4

from unified_animation import (
    TORSO_DEPTH_SIGN,
    _v,
    _norm,
    _build_basis,
    _world_to_basis,
    _basis_to_world,
)

# -----------------------------------------------------------------------
# Skeleton topology  (correct MediaPipe connections)
# -----------------------------------------------------------------------

# Toggles – set True to render face / leg landmarks and bones
SHOW_FACE = False
SHOW_LEGS = False

# Pose: face landmarks (0-10)
_POSE_FACE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]

# Pose: torso (always drawn)
_POSE_TORSO_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
]

# Pose: arms + wrist fingertip stubs (always drawn)
_POSE_ARM_CONNECTIONS = [
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
]

# Pose: legs (toggled by SHOW_LEGS)
_POSE_LEG_CONNECTIONS = [
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Assemble active pose connections
POSE_CONNECTIONS: list[tuple[int, int]] = list(_POSE_TORSO_CONNECTIONS) + list(_POSE_ARM_CONNECTIONS)
if SHOW_FACE:
    POSE_CONNECTIONS += _POSE_FACE_CONNECTIONS
if SHOW_LEGS:
    POSE_CONNECTIONS += _POSE_LEG_CONNECTIONS

# Landmark indices to draw (only torso/arm unless toggled)
_POSE_TORSO_ARM_INDICES = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
_POSE_FACE_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_POSE_LEG_INDICES = {25, 26, 27, 28, 29, 30, 31, 32}

POSE_VISIBLE_INDICES: set[int] = set(_POSE_TORSO_ARM_INDICES)
if SHOW_FACE:
    POSE_VISIBLE_INDICES |= _POSE_FACE_INDICES
if SHOW_LEGS:
    POSE_VISIBLE_INDICES |= _POSE_LEG_INDICES

# Hand connections (correct MediaPipe topology)
HAND_CONNECTIONS = [
    # Thumb (wrist → CMC → MCP → IP → tip)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger (wrist → MCP → PIP → DIP → tip)
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

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

        # Rig bone lengths (from rest-pose joint positions)
        self._rig_bone_len: dict[str, float] = {}
        for side in ("L", "R"):
            ua = self._jpos(f"FK-Upperarm.{side}")
            fa = self._jpos(f"FK-Forearm.{side}")
            hd = self._jpos(f"FK-Hand.{side}")
            self._rig_bone_len[f"Upperarm.{side}"] = (fa - ua).length()
            self._rig_bone_len[f"Forearm.{side}"] = (hd - fa).length()

        # Rig hand size (wrist → middle-finger MCP)
        self._rig_hand_size: dict[str, float] = {}
        for side in ("L", "R"):
            self._rig_hand_size[side] = self._hand_rig_size(side)

        # ---- camera save-state for 2D lock ----
        self._saved_cam_pos: Vec3 | None = None
        self._saved_cam_hpr: Vec3 | None = None
        self._cam_task_paused: bool = False

        # ---- 2D screen-space anchors (set when entering 2D mode) ----
        self._2d_screen_sc: Vec3 | None = None
        self._2d_screen_shoulder_w: float = 0.0

        # ---- HUD labels ----
        self._mode_text: OnscreenText | None = None
        self._controls_text: OnscreenText | None = None
        self._show_controls_hud()

        # ---- key binding ----
        self.accept(TOGGLE_KEY, self.toggle_mode)

    # ----- helpers --------------------------------------------------

    def _jpos(self, name: str) -> Vec3:
        return _v(self.actor.exposeJoint(None, PART_NAME, name).getPos(self.actor))

    def _hand_rig_size(self, side: str) -> float:
        try:
            w = self._jpos(f"DEF-Hand.{side}")
            m = self._jpos(f"DEF-Middle1.{side}")
            return (m - w).length()
        except Exception:
            return 0.05

    def _remap_dir(self, d: Vec3, cap_torso: tuple[Vec3, Vec3, Vec3]) -> Vec3:
        """Transform a direction from capture basis to rig basis."""
        local = _world_to_basis(d, cap_torso)
        local = Vec3(local.x, local.y, local.z * TORSO_DEPTH_SIGN)
        return _basis_to_world(local, self._rig_torso)

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
        elif self.mode == MODE_2D:
            self._2d_root.show()
            self._enter_2d_camera()

        self._refresh_mode_label()

    # ----- HUD -------------------------------------------------------

    def _show_controls_hud(self) -> None:
        """Always-visible controls hint (bottom-left)."""
        a2d = getattr(self.world, "aspect2d", None)
        if a2d is None:
            return
        try:
            aspect = float(self.world.getAspectRatio())
        except Exception:
            aspect = 1.0
        self._controls_text = OnscreenText(
            text="[V] Skeleton debug  |  [F] Eye tracking",
            parent=a2d,
            pos=(-aspect + 0.12, -0.95),
            scale=0.038,
            align=TextNode.ALeft,
            fg=(0.7, 0.7, 0.7, 0.8),
            bg=(0.06, 0.07, 0.09, 0.55),
            mayChange=False,
        )

    def _refresh_mode_label(self) -> None:
        if self._mode_text is not None:
            self._mode_text.destroy()
            self._mode_text = None
        if self.mode == MODE_OFF:
            return
        try:
            aspect = float(self.world.getAspectRatio())
        except Exception:
            aspect = 1.0
        self._mode_text = OnscreenText(
            text=f"Debug: {_MODE_LABELS[self.mode]}  [V] toggle",
            parent=getattr(self.world, "aspect2d", None) or self._2d_root,
            pos=(aspect - 0.12, 0.92),
            scale=0.042,
            align=TextNode.ARight,
            fg=(1.0, 1.0, 0.3, 0.9),
            bg=(0.08, 0.1, 0.14, 0.65),
            mayChange=False,
        )

    # ----- 2D camera lock / unlock -----------------------------------

    def _project_to_screen(self, actor_local_pos: Vec3) -> Vec3 | None:
        """Project an actor-local point to aspect2d coordinates."""
        cam = getattr(self.world, "camera", None) or getattr(self.world, "cam", None)
        lens = getattr(self.world, "camLens", None)
        render = getattr(self.world, "render", None)
        if cam is None or lens is None or render is None:
            return None
        world_pos = render.getRelativePoint(self.actor, LPoint3f(*actor_local_pos))
        cam_pos = cam.getRelativePoint(render, world_pos)
        p2d = LPoint2f()
        if not lens.project(LPoint3f(*cam_pos), p2d):
            return None
        try:
            aspect = float(self.world.getAspectRatio())
        except Exception:
            aspect = 1.0
        return Vec3(p2d.x * aspect, 0, p2d.y)

    def _cache_2d_anchors(self) -> None:
        """Project rig shoulders to screen space and cache for 2D overlay."""
        ls_s = self._project_to_screen(self._rig_ls)
        rs_s = self._project_to_screen(self._rig_rs)
        if ls_s is not None and rs_s is not None:
            self._2d_screen_sc = (ls_s + rs_s) * 0.5
            self._2d_screen_shoulder_w = abs(rs_s.x - ls_s.x)
        else:
            self._2d_screen_sc = None
            self._2d_screen_shoulder_w = 0.0

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

        # Cache screen-space anchors for the 2D overlay
        self._cache_2d_anchors()

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

    def _build_capture_torso(self, pose_lms):
        """Return capture torso basis or None."""
        ls = pose_lms.get(11)
        rs = pose_lms.get(12)
        if ls is None or rs is None:
            return None
        lh = pose_lms.get(23)
        rh = pose_lms.get(24)
        if lh is not None and rh is not None:
            cap_hc = (lh + rh) * 0.5
            cap_sc = (ls + rs) * 0.5
            basis = _build_basis(cap_sc - cap_hc, rs - ls)
        else:
            # No hips — use a vertical-ish fallback
            basis = _build_basis(Vec3(0, 1, 0), rs - ls)
        return basis

    def _update_3d(self, pose_lms, hand_lms):
        self._clear_node("_3d_geom")

        cap_torso = self._build_capture_torso(pose_lms)
        if cap_torso is None:
            return

        segs = LineSegs("lm-debug-3d")

        # --- build bone-length-matched pose landmarks ---
        rig_pose: dict[int, Vec3] = {}

        # Pin torso anchors to rig positions
        rig_pose[11] = Vec3(self._rig_ls)
        rig_pose[12] = Vec3(self._rig_rs)
        rig_pose[23] = Vec3(self._rig_lh)
        rig_pose[24] = Vec3(self._rig_rh)

        # Arm chains — use capture direction, rig bone length
        for side, sh_idx, el_idx, wr_idx in [("L", 11, 13, 15), ("R", 12, 14, 16)]:
            sh_cap = pose_lms.get(sh_idx)
            el_cap = pose_lms.get(el_idx)
            wr_cap = pose_lms.get(wr_idx)
            if sh_cap is not None and el_cap is not None:
                ua_dir_cap = _norm(el_cap - sh_cap)
                if ua_dir_cap is not None:
                    ua_dir_rig = _norm(self._remap_dir(ua_dir_cap, cap_torso))
                    if ua_dir_rig is not None:
                        rig_pose[el_idx] = rig_pose[sh_idx] + ua_dir_rig * self._rig_bone_len[f"Upperarm.{side}"]

                        if wr_cap is not None:
                            fa_dir_cap = _norm(wr_cap - el_cap)
                            if fa_dir_cap is not None:
                                fa_dir_rig = _norm(self._remap_dir(fa_dir_cap, cap_torso))
                                if fa_dir_rig is not None:
                                    rig_pose[wr_idx] = rig_pose[el_idx] + fa_dir_rig * self._rig_bone_len[f"Forearm.{side}"]

        # Wrist fingertip stubs (17-22) — small offsets from rig wrist
        # These are minor; place them near the wrist with direction hint
        cap_sc = ((pose_lms.get(11) or Vec3()) + (pose_lms.get(12) or Vec3())) * 0.5
        cap_sd = ((pose_lms.get(12) or Vec3()) - (pose_lms.get(11) or Vec3())).length() or 1.0
        stub_scale = self._rig_shoulder_dist / cap_sd
        for idx in (17, 18, 19, 20, 21, 22):
            p = pose_lms.get(idx)
            wr_idx = 15 if idx in (17, 19, 21) else 16
            wr_cap = pose_lms.get(wr_idx)
            rig_wr = rig_pose.get(wr_idx)
            if p is not None and wr_cap is not None and rig_wr is not None:
                offset_cap = p - wr_cap
                offset_rig = self._remap_dir(offset_cap, cap_torso) * stub_scale
                rig_pose[idx] = rig_wr + offset_rig

        # Filter to visible indices
        visible_pose = {i: p for i, p in rig_pose.items() if i in POSE_VISIBLE_INDICES}

        _draw_bones(segs, visible_pose, POSE_CONNECTIONS, POSE_BONE_COLOR, 2.5)
        _draw_markers_3d(segs, visible_pose, POSE_MARKER_COLOR, 5.0, 0.008)

        # --- hands: anchored at rig wrist, scaled to rig hand size ---
        for side, hlms in hand_lms.items():
            if hlms is None:
                continue
            color = LEFT_HAND_COLOR if side == "L" else RIGHT_HAND_COLOR
            wr_idx = 15 if side == "L" else 16
            rig_wrist = rig_pose.get(wr_idx)
            if rig_wrist is None:
                continue

            w0 = hlms.get(0)
            if w0 is None:
                continue

            mid = hlms.get(9)
            cap_hand_span = (mid - w0).length() if mid is not None else 0.15
            rig_hand_span = self._rig_hand_size.get(side, 0.05)
            hand_scale = rig_hand_span / max(cap_hand_span, 1e-4)

            if self.hand_world_space:
                # ASLLVD: hand landmarks in same capture space — remap direction + scale
                rig_hand: dict[int, Vec3] = {}
                for i, p in hlms.items():
                    offset = p - w0
                    rig_offset = self._remap_dir(offset, cap_torso) * hand_scale
                    rig_hand[i] = rig_wrist + rig_offset
            else:
                # CSV: image-normalised space — translate + scale relative to wrist
                rig_hand = {i: rig_wrist + (p - w0) * hand_scale for i, p in hlms.items()}

            _draw_bones(segs, rig_hand, HAND_CONNECTIONS, color, 1.5)
            _draw_markers_3d(segs, rig_hand, color, 3.5, 0.005)

        node = segs.create()
        self._3d_geom = self._3d_root.attachNewNode(node)

    # =================================================================
    #  2D overlay — scaled to match model on screen
    # =================================================================

    def _update_2d(self, pose_lms, hand_lms):
        self._clear_node("_2d_geom")
        if not pose_lms:
            return

        ls_cap = pose_lms.get(11)
        rs_cap = pose_lms.get(12)
        if ls_cap is None or rs_cap is None:
            return

        # Capture shoulder stats (X = horizontal, Y = vertical in our space)
        cap_sc_x = (ls_cap.x + rs_cap.x) * 0.5
        cap_sc_y = (ls_cap.y + rs_cap.y) * 0.5
        cap_sd = abs(rs_cap.x - ls_cap.x)
        if cap_sd < 1e-6:
            return

        # Screen anchor from projection (cached when entering 2D mode)
        if self._2d_screen_sc is None or self._2d_screen_shoulder_w < 1e-6:
            return
        scr_sc = self._2d_screen_sc
        s = self._2d_screen_shoulder_w / cap_sd

        def _to2d(p: Vec3) -> Vec3:
            return Vec3(
                scr_sc.x + (p.x - cap_sc_x) * s,
                0,
                scr_sc.z + (p.y - cap_sc_y) * s,
            )

        segs = LineSegs("lm-debug-2d")

        # --- pose (filtered to visible indices) ---
        sp = {i: _to2d(p) for i, p in pose_lms.items() if i in POSE_VISIBLE_INDICES}
        _draw_bones(segs, sp, POSE_CONNECTIONS, POSE_BONE_COLOR, 2.5)
        _draw_markers_2d(segs, sp, POSE_MARKER_COLOR, 6.0, 0.018)

        # --- hands ---
        for side, hlms in hand_lms.items():
            if hlms is None:
                continue
            color = LEFT_HAND_COLOR if side == "L" else RIGHT_HAND_COLOR

            if self.hand_world_space:
                sh = {i: _to2d(p) for i, p in hlms.items()}
            else:
                # CSV: anchor at the projected pose-wrist position
                wi = 15 if side == "L" else 16
                pw = pose_lms.get(wi)
                w0 = hlms.get(0)
                if pw is None or w0 is None:
                    continue
                sw = _to2d(pw)
                mid = hlms.get(9)
                cap_hs = (mid - w0).length() if mid is not None else 0.15
                # Target hand screen size ≈ 35% of screen shoulder width
                hs = (self._2d_screen_shoulder_w * 0.35 / max(cap_hs, 1e-4))
                sh = {
                    i: Vec3(sw.x + (p.x - w0.x) * hs, 0, sw.z + (p.y - w0.y) * hs)
                    for i, p in hlms.items()
                }

            _draw_bones(segs, sh, HAND_CONNECTIONS, color, 1.5)
            _draw_markers_2d(segs, sh, color, 4.0, 0.010)

        node = segs.create()
        self._2d_geom = self._2d_root.attachNewNode(node)

    # ----- cleanup ---------------------------------------------------

    def destroy(self) -> None:
        self.ignoreAll()
        self._clear_node("_3d_geom")
        self._clear_node("_2d_geom")
        if self._mode_text is not None:
            self._mode_text.destroy()
        if self._controls_text is not None:
            self._controls_text.destroy()
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
