import math
import os

from direct.actor.Actor import Actor
from panda3d.core import (
    LPoint3f,
    Vec3,
    Filename,
    AmbientLight,
    DirectionalLight,
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain.bam.pz")

# Default anim subfolder inside `anim/` (change this in one place)
ANIM_SUBFOLDER = "act"

# Default transform applied to the loaded character
POS = LPoint3f(0, 0, -1)
HPR = Vec3(0, -90, 0)
SCALE = 3.0


def model_filename() -> Filename:
    return Filename.fromOsSpecific(MODEL_PATH)


def load_actor(world, pos: LPoint3f = POS, hpr: Vec3 = HPR, scale: float = SCALE) -> Actor:
    """Load the model via the world's loader, parent to the world's render,
    and apply the default transform. Raises FileNotFoundError on load failure.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\n\nPlease place the model file at: {MODEL_PATH}")

    model_fname = model_filename()
    model_np = world.loader.loadModel(model_fname)
    if model_np is None or model_np.isEmpty():
        raise FileNotFoundError(f"Panda failed to load model: {MODEL_PATH}")

    character = Actor(model_np)
    character.reparentTo(world.render)
    character.setPos(pos)
    character.setHpr(hpr)
    character.setScale(scale)

    return character


def setup_lighting(world) -> None:
    ambient = AmbientLight("ambient-light")
    ambient.setColor((0.52, 0.52, 0.58, 1))
    ambient_np = world.render.attachNewNode(ambient)
    world.render.setLight(ambient_np)

    key = DirectionalLight("key-light")
    key.setColor((0.75, 0.75, 0.78, 1))
    key_np = world.render.attachNewNode(key)
    key_np.setHpr(-20, -18, 0)
    world.render.setLight(key_np)


def frame_camera(world, actor) -> None:
    """Frame the world's camera to the given actor. Safe to call when
    a camera/lens may be missing; failures are swallowed.
    """
    try:
        bounds = actor.getTightBounds()
        if not bounds or bounds[0] is None or bounds[1] is None:
            if hasattr(world, "cam"):
                world.cam.setPos(0, -12, 1.5)
                world.cam.lookAt(0, 0, 1.5)
            return

        min_point, max_point = bounds
        center = (min_point + max_point) * 0.5
        size = max_point - min_point

        # Prefer the standard ShowBase camLens attribute if present
        cam_lens = getattr(world, "camLens", None)
        if cam_lens is None:
            # try to fetch lens from cam node if available
            try:
                cam_node = getattr(world, "cam", None)
                if cam_node is not None and getattr(cam_node, "node", None) is not None:
                    cam_lens = cam_node.node().getLens()
            except Exception:
                cam_lens = None

        if cam_lens is None:
            return

        # set lens params
        try:
            cam_lens.setFov(50)
            cam_lens.setNearFar(0.1, 1000)
            horizontal_fov, vertical_fov = cam_lens.getFov()
        except Exception:
            # fallback to no framing
            return

        half_width = max(size.x * 0.5, 1.0)
        half_height = max(size.z * 0.5, 1.0)

        distance_for_width = half_width / math.tan(math.radians(horizontal_fov * 0.5))
        distance_for_height = half_height / math.tan(math.radians(vertical_fov * 0.5))
        camera_distance = max(distance_for_width, distance_for_height) * 1.15

        focus_point = LPoint3f(center.x, center.y, center.z + size.z * 0.05)
        camera_point = LPoint3f(center.x, center.y - camera_distance, center.z + size.z * 0.02)

        # Prefer setting transform on `camera` NodePath (ShowBase.camera), fall back to `cam`
        camera_node = getattr(world, "camera", None) or getattr(world, "cam", None)
        if camera_node is not None:
            try:
                camera_node.setPos(camera_point)
                camera_node.lookAt(focus_point)
            except Exception:
                pass
    except Exception:
        # Best-effort only; do not raise
        pass


def get_anim_dir(subfolder: str | None = None) -> str:
    """Return the absolute path to the chosen anim subfolder inside `anim/`.
    Use the module-level `ANIM_SUBFOLDER` by default.
    """
    sub = subfolder if subfolder else ANIM_SUBFOLDER
    return os.path.abspath(os.path.join(BASE_DIR, "anim", sub))
