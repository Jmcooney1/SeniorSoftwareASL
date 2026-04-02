from direct.showbase.DirectObject import DirectObject
from panda3d.core import ClockObject, Quat, Vec3, WindowProperties


class FlyCameraController(DirectObject):
    def __init__(
        self,
        base,
        camera,
        move_speed: float = 10.0,
        look_sensitivity_degrees: float = 0.1,
    ) -> None:
        super().__init__()
        self.base = base
        self.camera = camera
        self.move_speed = move_speed
        self.look_sensitivity_degrees = look_sensitivity_degrees

        initial_hpr = self.camera.getHpr(self.base.render)
        self.yaw_degrees = initial_hpr.x
        self.pitch_degrees = initial_hpr.y
        self.last_pointer = None
        self.suppress_next_mouse_delta = False

        self.key_map = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }

        self.accept("w", self._set_key, ["forward", True])
        self.accept("w-up", self._set_key, ["forward", False])
        self.accept("s", self._set_key, ["backward", True])
        self.accept("s-up", self._set_key, ["backward", False])
        self.accept("a", self._set_key, ["left", True])
        self.accept("a-up", self._set_key, ["left", False])
        self.accept("d", self._set_key, ["right", True])
        self.accept("d-up", self._set_key, ["right", False])
        self.accept("space", self._set_key, ["up", True])
        self.accept("space-up", self._set_key, ["up", False])
        self.accept("control", self._set_key, ["down", True])
        self.accept("control-up", self._set_key, ["down", False])
        self.accept("lcontrol", self._set_key, ["down", True])
        self.accept("lcontrol-up", self._set_key, ["down", False])
        self.accept("rcontrol", self._set_key, ["down", True])
        self.accept("rcontrol-up", self._set_key, ["down", False])

        self._capture_mouse()
        self._apply_orientation()
        self.base.taskMgr.add(self._update, "fly-camera-controller")

    def destroy(self) -> None:
        self._release_mouse()
        self.ignoreAll()
        self.base.taskMgr.remove("fly-camera-controller")

    def _set_key(self, key: str, is_down: bool) -> None:
        self.key_map[key] = is_down

    def _capture_mouse(self) -> None:
        if not hasattr(self.base.win, "requestProperties"):
            return
        props = WindowProperties()
        props.setCursorHidden(True)
        if hasattr(props, "setMouseMode"):
            props.setMouseMode(WindowProperties.M_confined)
        self.base.win.requestProperties(props)
        self._center_pointer()

    def _release_mouse(self) -> None:
        if not hasattr(self.base.win, "requestProperties"):
            return
        props = WindowProperties()
        props.setCursorHidden(False)
        if hasattr(props, "setMouseMode"):
            props.setMouseMode(WindowProperties.M_absolute)
        self.base.win.requestProperties(props)
        self.last_pointer = None
        self.suppress_next_mouse_delta = False

    def _update(self, task):
        dt = ClockObject.getGlobalClock().getDt()
        self._rotate_camera_from_mouse()
        self._move_camera(dt)
        return task.cont

    def _center_pointer(self) -> None:
        if not hasattr(self.base.win, "movePointer"):
            return
        center_x = self.base.win.getXSize() // 2
        center_y = self.base.win.getYSize() // 2
        self.base.win.movePointer(0, center_x, center_y)
        self.last_pointer = (center_x, center_y)
        self.suppress_next_mouse_delta = True

    def _rotate_camera_from_mouse(self) -> None:
        if not self.base.mouseWatcherNode.hasMouse() or not hasattr(self.base.win, "getPointer"):
            self.last_pointer = None
            return

        pointer = self.base.win.getPointer(0)
        pointer_xy = (pointer.getX(), pointer.getY())

        if self.last_pointer is None:
            self.last_pointer = pointer_xy
            return

        if self.suppress_next_mouse_delta:
            self.last_pointer = pointer_xy
            self.suppress_next_mouse_delta = False
            return

        delta_x = pointer_xy[0] - self.last_pointer[0]
        delta_y = pointer_xy[1] - self.last_pointer[1]
        self.last_pointer = pointer_xy

        if abs(delta_x) > 200 or abs(delta_y) > 200:
            return

        if delta_x == 0 and delta_y == 0:
            return

        self.yaw_degrees -= delta_x * self.look_sensitivity_degrees
        self.pitch_degrees = max(
            -89.5,
            min(89.5, self.pitch_degrees - (delta_y * self.look_sensitivity_degrees)),
        )
        self._apply_orientation()

        if self._pointer_near_window_edge(pointer_xy):
            self._center_pointer()

    def _pointer_near_window_edge(self, pointer_xy) -> bool:
        margin = 32
        x_size = self.base.win.getXSize()
        y_size = self.base.win.getYSize()
        x, y = pointer_xy
        return x <= margin or x >= (x_size - margin) or y <= margin or y >= (y_size - margin)

    def _move_camera(self, dt: float) -> None:
        camera_quat = self.camera.getQuat(self.base.render)
        forward = camera_quat.xform(Vec3(0, 1, 0))
        right = camera_quat.xform(Vec3(1, 0, 0))
        up = Vec3(0, 0, 1)

        velocity = Vec3(0, 0, 0)
        if self.key_map["forward"]:
            velocity += forward
        if self.key_map["backward"]:
            velocity -= forward
        if self.key_map["left"]:
            velocity -= right
        if self.key_map["right"]:
            velocity += right
        if self.key_map["up"]:
            velocity += up
        if self.key_map["down"]:
            velocity -= up

        if velocity.length_squared() == 0:
            return

        velocity.normalize()
        self.camera.setPos(self.camera.getPos() + (velocity * self.move_speed * dt))

    def _apply_orientation(self) -> None:
        orientation = Quat()
        orientation.setHpr((self.yaw_degrees, self.pitch_degrees, 0.0))
        orientation.normalize()
        self.camera.setQuat(self.base.render, orientation)
