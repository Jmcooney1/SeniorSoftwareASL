"""Landmark debug overlay for ASL sign animation.

Toggle with **V**:  OFF → 3D (always-on-top) → 2D (flat overlay) → OFF

3D mode
    Landmarks and bones rendered on top of the character in rig world-space
    so you can orbit freely and inspect the skeleton from any angle.

2D mode
    A flat MediaPipe-style skeleton drawn on ``aspect2d``.  The camera
    locks to a front view and zooms out slightly so the overlay aligns
    with the model beneath it.
"""

from __future__ import annotations

import math

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from panda3d.core import LineSegs, NodePath, TextNode, Vec3, Vec4

from unified_animation import (
    TORSO_DEPTH_SIGN,
    _v,
    _norm,
    _build_basis,
    _world_to_basis,
    _basis_to_world,
)

# -----------------------------------------------------------------------
# Skeleton topology
# -----------------------------------------------------------------------

POSE_CONNECTIONS = [
    (11, 12),   # shoulders
    (11, 13),   # L shoulder → L elbow
    (13, 15),   # L elbow → L wrist
    (12, 14),   # R shoulder → R elbow
    (14, 16),   # R elbow → R wrist
    (11, 23),   # L shoulder → L hip
    (12, 24),   # R shoulder → R hip
    (23, 24),   # hips
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
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
        self._rig_hand_size = {
            s: self._hand_rig_size(s) for s in ("L", "R")
        }

        # ---- camera save-state for 2D lock ----
        self._saved_cam_pos: Vec3 | None = None
        self._saved_cam_hpr: Vec3 | None = None
        self._cam_task_paused: bool = False

        # ---- HUD label ----
        self._mode_text: OnscreenText | None = None

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

        self._refresh_label()

    # ----- HUD label -------------------------------------------------

    def _refresh_label(self) -> None:
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
        cam.setPos(CAMERA_TARGET_X, CAMERA_TARGET_Y - 2.2, 0.35)
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
    #  3D overlay
    # =================================================================

    def _clear_node(self, attr: str) -> None:
        node = getattr(self, attr, None)
        if node is not None:
            node.removeNode()
        setattr(self, attr, None)

    def _capture_affine(self, pose_lms):
        """Return (cap_center, cap_torso, scale) or None."""
        ls = pose_lms.get(11)
        rs = pose_lms.get(12)
        lh = pose_lms.get(23)
        rh = pose_lms.get(24)
        if ls is None or rs is None or lh is None or rh is None:
            return None
        cap_sc = (ls + rs) * 0.5
        cap_hc = (lh + rh) * 0.5
        cap_torso = _build_basis(cap_sc - cap_hc, rs - ls)
        if cap_torso is None:
            return None
        cap_sd = (rs - ls).length()
        if cap_sd < 1e-6:
            return None
        scale = self._rig_shoulder_dist / cap_sd
        return cap_sc, cap_torso, scale

    def _to_rig(self, pos, cap_sc, cap_torso, scale):
        """Map a capture-space position → actor-local rig position."""
        offset = pos - cap_sc
        local = _world_to_basis(offset, cap_torso)
        local = Vec3(local.x, local.y, local.z * TORSO_DEPTH_SIGN)
        return self._rig_sc + _basis_to_world(local, self._rig_torso) * scale

    def _anchor_hand_3d(self, hlms, rig_wrist, side):
        """Re-anchor CSV hand landmarks at *rig_wrist*."""
        w0 = hlms.get(0)
        if w0 is None:
            return {}
        mid = hlms.get(9)
        cap_hs = (mid - w0).length() if mid is not None else 0.15
        rig_hs = self._rig_hand_size.get(side, 0.05)
        s = rig_hs / max(cap_hs, 1e-4)
        return {i: rig_wrist + (p - w0) * s for i, p in hlms.items()}

    def _update_3d(self, pose_lms, hand_lms):
        self._clear_node("_3d_geom")
        affine = self._capture_affine(pose_lms)
        if affine is None:
            return
        cap_sc, cap_torso, scale = affine

        ls = LineSegs("lm-debug-3d")

        # --- pose ---
        rig_pose: dict[int, Vec3] = {
            i: self._to_rig(p, cap_sc, cap_torso, scale)
            for i, p in pose_lms.items()
        }
        _draw_bones(ls, rig_pose, POSE_CONNECTIONS, POSE_BONE_COLOR, 2.5)
        _draw_markers_3d(ls, rig_pose, POSE_MARKER_COLOR, 5.0, 0.008)

        # --- hands ---
        for side, hlms in hand_lms.items():
            if hlms is None:
                continue
            color = LEFT_HAND_COLOR if side == "L" else RIGHT_HAND_COLOR

            if self.hand_world_space:
                rig_hand = {
                    i: self._to_rig(p, cap_sc, cap_torso, scale)
                    for i, p in hlms.items()
                }
            else:
                wi = 15 if side == "L" else 16
                rw = rig_pose.get(wi)
                if rw is None:
                    continue
                rig_hand = self._anchor_hand_3d(hlms, rw, side)

            _draw_bones(ls, rig_hand, HAND_CONNECTIONS, color, 1.5)
            _draw_markers_3d(ls, rig_hand, color, 3.5, 0.005)

        node = ls.create()
        self._3d_geom = self._3d_root.attachNewNode(node)

    # =================================================================
    #  2D overlay
    # =================================================================

    def _update_2d(self, pose_lms, hand_lms):
        self._clear_node("_2d_geom")
        if not pose_lms:
            return

        # Compute screen-space transform (capture XY → aspect2d coords)
        xs = [p.x for p in pose_lms.values()]
        ys = [p.y for p in pose_lms.values()]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.01)
        s = 1.5 / span       # scale body to ~1.5 units height
        oy = -0.05           # slight vertical offset to centre on model

        def _to2d(p: Vec3) -> Vec3:
            return Vec3((p.x - cx) * s, 0, (p.y - cy) * s + oy)

        ls = LineSegs("lm-debug-2d")

        # --- pose ---
        sp = {i: _to2d(p) for i, p in pose_lms.items()}
        _draw_bones(ls, sp, POSE_CONNECTIONS, POSE_BONE_COLOR, 2.5)
        _draw_markers_2d(ls, sp, POSE_MARKER_COLOR, 6.0, 0.018)

        # --- hands ---
        for side, hlms in hand_lms.items():
            if hlms is None:
                continue
            color = LEFT_HAND_COLOR if side == "L" else RIGHT_HAND_COLOR

            if self.hand_world_space:
                sh = {i: _to2d(p) for i, p in hlms.items()}
            else:
                # anchor at pose wrist screen position
                wi = 15 if side == "L" else 16
                pw = pose_lms.get(wi)
                w0 = hlms.get(0)
                if pw is None or w0 is None:
                    continue
                sw = _to2d(pw)
                mid = hlms.get(9)
                cap_hs = (mid - w0).length() if mid is not None else 0.15
                # target hand screen-size ≈ 35 % of shoulder width
                ls_p = pose_lms.get(11)
                rs_p = pose_lms.get(12)
                shoulder_w = (rs_p - ls_p).length() if ls_p is not None and rs_p is not None else 0.3
                hs = (shoulder_w * 0.35 / max(cap_hs, 1e-4)) * s
                sh = {
                    i: Vec3(sw.x + (p.x - w0.x) * hs, 0, sw.z + (p.y - w0.y) * hs)
                    for i, p in hlms.items()
                }

            _draw_bones(ls, sh, HAND_CONNECTIONS, color, 1.5)
            _draw_markers_2d(ls, sh, color, 4.0, 0.010)

        node = ls.create()
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
