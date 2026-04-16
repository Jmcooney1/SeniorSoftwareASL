from __future__ import annotations

import math

from direct.showbase.DirectObject import DirectObject
from panda3d.core import ClockObject, LPoint3f


class CameraController(DirectObject):
    def __init__(
        self,
        base,
        camera,
        orbit_center: LPoint3f | tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 5.2,
        camera_height: float = 1.82,
        initial_azimuth_degrees: float = 0.0,
        orbit_speed_degrees: float = 90.0,
        zoom_speed: float = 3.0,
        min_distance: float = 2.4,
        max_distance: float = 8.5,
    ) -> None:
        super().__init__()
        self.base = base
        self.camera = camera
        self.orbit_center = self._coerce_point(orbit_center)
        self.distance = distance
        self.camera_height = camera_height
        self.azimuth_degrees = initial_azimuth_degrees
        self.orbit_speed_degrees = orbit_speed_degrees
        self.zoom_speed = zoom_speed
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.key_map = {
            "orbit_left": False,
            "orbit_right": False,
            "zoom_in": False,
            "zoom_out": False,
        }

        self._bind_key("a", "orbit_left")
        self._bind_key("arrow_left", "orbit_left")
        self._bind_key("d", "orbit_right")
        self._bind_key("arrow_right", "orbit_right")
        self._bind_key("w", "zoom_in")
        self._bind_key("arrow_up", "zoom_in")
        self._bind_key("s", "zoom_out")
        self._bind_key("arrow_down", "zoom_out")

        self._apply_camera_transform()
        self.base.taskMgr.add(self._update, "orbit-camera-controller")

    def destroy(self) -> None:
        self.ignoreAll()
        self.base.taskMgr.remove("orbit-camera-controller")

    def set_camera_height(self, height: float) -> None:
        self.camera_height = height
        self._apply_camera_transform()

    def set_distance(self, distance: float) -> None:
        self.distance = self._clamp_distance(distance)
        self._apply_camera_transform()

    def set_orbit_center(self, orbit_center: LPoint3f | tuple[float, float, float]) -> None:
        self.orbit_center = self._coerce_point(orbit_center)
        self._apply_camera_transform()

    def _bind_key(self, key: str, action: str) -> None:
        self.accept(key, self._set_key, [action, True])
        self.accept(f"{key}-up", self._set_key, [action, False])

    def _set_key(self, key: str, is_down: bool) -> None:
        self.key_map[key] = is_down

    def _update(self, task):
        dt = ClockObject.getGlobalClock().getDt()
        did_move = False

        if self.key_map["orbit_left"]:
            self.azimuth_degrees -= self.orbit_speed_degrees * dt
            did_move = True
        if self.key_map["orbit_right"]:
            self.azimuth_degrees += self.orbit_speed_degrees * dt
            did_move = True
        if self.key_map["zoom_in"]:
            self.distance = self._clamp_distance(self.distance - (self.zoom_speed * dt))
            did_move = True
        if self.key_map["zoom_out"]:
            self.distance = self._clamp_distance(self.distance + (self.zoom_speed * dt))
            did_move = True

        if did_move:
            self._apply_camera_transform()

        return task.cont

    def _apply_camera_transform(self) -> None:
        azimuth_radians = math.radians(self.azimuth_degrees)
        camera_x = self.orbit_center.x + (math.sin(azimuth_radians) * self.distance)
        camera_y = self.orbit_center.y - (math.cos(azimuth_radians) * self.distance)
        self.camera.setPos(camera_x, camera_y, self.camera_height)
        self.camera.lookAt(self.orbit_center)

    def _clamp_distance(self, distance: float) -> float:
        return max(self.min_distance, min(self.max_distance, distance))

    @staticmethod
    def _coerce_point(point: LPoint3f | tuple[float, float, float]) -> LPoint3f:
        if isinstance(point, LPoint3f):
            return LPoint3f(point)
        return LPoint3f(*point)


OrbitCameraController = CameraController
