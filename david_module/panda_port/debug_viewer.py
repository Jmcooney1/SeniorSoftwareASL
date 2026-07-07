"""Animation debugging mini-app.

Replaces the old ``character_app.py`` launcher.  Opens a single Panda3D
window and lets you cycle through a hard-coded list of TEST SIGNS (updated
by hand each debugging iteration) plus browse/jump through the FULL sign
catalogue without going through the main app's library/search flow.

Run:
    python david_module/panda_port/debug_viewer.py

Controls
--------
    N / P           next / previous TEST sign (the hard-coded debug set)
    ] / [           next / previous sign in the full catalogue
    Page Up/Down    jump ±25 through the catalogue
    (click)         load a sign from the on-screen catalogue list
    search box      type a prefix, press Enter → jump to that part of the
                    catalogue (first-letter jump works: e.g. "m" + Enter)
    + / -           playback speed up / down (for odd-framerate clips)
    R               restart the current clip from frame 0
    V               cycle skeleton overlay: OFF → 3D → 2D (landmark_debug)
    mouse drag      orbit camera (existing camera controller)
    ESC             quit
"""

from __future__ import annotations

import bisect
import os
import sys
import types
from pathlib import Path

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from direct.gui.DirectGui import DirectButton, DirectEntry
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, loadPrcFileData

from animation import CSV_FILENAME_RE, CSVRigAnimator, CSVSignClip
import panda_core
from landmark_debug import LandmarkVisualizer

# ---------------------------------------------------------------------------
# Hard-coded test set — update this list each debugging iteration.
# Names must match the sign name inside the CSV filename (case-insensitive).
# ---------------------------------------------------------------------------

TEST_SIGNS = [
    "Binoculars",        # two-handed; palm-inward + flexion/extension test
    "Choke",             # one-handed; palm-toward-chest + idle-arm tuck test
    "Good, Thank You",   # one-handed; palm-to-chest tilt + thumb test
    "Ironing",           # flat palm-down hold test
    "Acquire",           # left-thumb basis-flip regression test (fix 18)
    "AllNight",          # left-thumb + arm-jitter test (fix 19)
]

# Full catalogue folder (all ~4k signs), with fallback to the demo subset.
REPO_ROOT = Path(HERE).resolve().parent.parent
CATALOGUE_DIRS = [
    REPO_ROOT / "dataSet" / "david_dataset" / "landmarks",
    REPO_ROOT / "dataSet" / "david_dataset" / "best",
]

PLAYBACK_RATES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LIST_ROWS = 11  # entries shown in the on-screen catalogue list


def _scan_catalogue() -> list[tuple[str, Path]]:
    for d in CATALOGUE_DIRS:
        if not d.is_dir():
            continue
        out = []
        for p in d.iterdir():
            m = CSV_FILENAME_RE.match(p.name)
            if m:
                out.append((m.group(1), p))
        if out:
            out.sort(key=lambda t: t[0].lower())
            return out
    return []


class DebugViewer:
    def __init__(self) -> None:
        loadPrcFileData("", "win-size 1200 1000")
        loadPrcFileData("", "window-title ASL Animation Debug Viewer")
        self.base = ShowBase()
        self.base.disableMouse()
        self.base.setBackgroundColor(0.08, 0.09, 0.11, 1)

        self.catalogue = _scan_catalogue()
        self.names_lower = [n.lower() for n, _ in self.catalogue]
        if not self.catalogue:
            raise SystemExit(f"No sign CSVs found in: {CATALOGUE_DIRS}")

        # Resolve the test set against the catalogue (skip missing quietly
        # but say so on the console).
        self.test_indices: list[int] = []
        for name in TEST_SIGNS:
            try:
                self.test_indices.append(self.names_lower.index(name.lower()))
            except ValueError:
                print(f"[debug_viewer] test sign not in catalogue: {name!r}")
        self.test_pos = 0
        self.cat_index = self.test_indices[0] if self.test_indices else 0

        # ---- scene ----
        self.character = panda_core.load_actor(self.base)
        panda_core.setup_lighting(self.base)
        try:
            panda_core.frame_camera(self.base, self.character)
        except Exception:
            pass
        cam = getattr(self.base, "camera", None) or getattr(self.base, "cam", None)
        try:
            self.camera_ctrl = panda_core.create_camera_controller(self.base, cam) if cam else None
        except Exception:
            self.camera_ctrl = None

        self.animator = CSVRigAnimator(self.character)

        # Skeleton overlay (its own V-key binding lives in LandmarkVisualizer)
        self.viz = LandmarkVisualizer(
            self.base, self.character,
            camera_controller=self.camera_ctrl,
            hand_world_space=getattr(self.animator, "hand_world_space", False),
        )

        # Character pose controller — applies the non-animation cosmetic
        # pose (ponytail droop, eye tracking).  Without it the ponytail
        # sticks out rigidly horizontal at its rest orientation.
        try:
            self.pose_ctrl = panda_core.create_character_pose_controller(
                self.base, self.character, camera=cam,
            )
        except Exception:
            self.pose_ctrl = None

        # ---- playback state ----
        self.rate_i = PLAYBACK_RATES.index(1.0)
        self._time_origin = 0.0
        self._last_real_time = 0.0
        self.base.taskMgr.add(self._tick, "debug-viewer-tick")

        # ---- UI ----
        self._search_focused = False
        self._build_ui()
        self._bind_keys()

        self._load_index(self.cat_index)

    # ------------------------------------------------------------------
    # Playback task: feeds the animator a rate-scaled, origin-shifted time
    # so R restarts cleanly and +/- change speed without skipping.
    # ------------------------------------------------------------------

    def _tick(self, task):
        self._last_real_time = task.time
        scaled = types.SimpleNamespace(
            time=(task.time - self._time_origin) * PLAYBACK_RATES[self.rate_i],
            cont=task.cont,
        )
        self.animator.update(scaled)
        self.viz.update(self.animator.last_pose_lms, self.animator.last_hand_lms)
        return task.cont

    # ------------------------------------------------------------------
    # Sign loading / navigation
    # ------------------------------------------------------------------

    def _load_index(self, idx: int) -> None:
        idx = max(0, min(len(self.catalogue) - 1, idx))
        self.cat_index = idx
        name, path = self.catalogue[idx]
        self.animator.set_clip(CSVSignClip(path))
        self._time_origin = self._last_real_time
        self._refresh_ui()

    def _next_test(self, step: int) -> None:
        if not self.test_indices:
            return
        self.test_pos = (self.test_pos + step) % len(self.test_indices)
        self._load_index(self.test_indices[self.test_pos])

    def _step_catalogue(self, step: int) -> None:
        self._load_index(self.cat_index + step)

    def _jump_prefix(self, text: str) -> None:
        text = text.strip().lower()
        if not text:
            return
        idx = bisect.bisect_left(self.names_lower, text)
        self._load_index(idx)

    def _change_rate(self, step: int) -> None:
        # Re-anchor the origin so the clip position doesn't jump when the
        # rate changes.
        cur = (self._last_real_time - self._time_origin) * PLAYBACK_RATES[self.rate_i]
        self.rate_i = max(0, min(len(PLAYBACK_RATES) - 1, self.rate_i + step))
        new_rate = PLAYBACK_RATES[self.rate_i]
        self._time_origin = self._last_real_time - (cur / new_rate if new_rate else 0.0)
        self._refresh_ui()

    def _restart(self) -> None:
        self.animator.set_clip(CSVSignClip(self.catalogue[self.cat_index][1]))
        self._time_origin = self._last_real_time
        self._refresh_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        a2d = self.base.aspect2d
        try:
            aspect = float(self.base.getAspectRatio())
        except Exception:
            aspect = 1.2

        self._status = OnscreenText(
            parent=a2d, pos=(-aspect + 0.05, 0.95), scale=0.042,
            align=TextNode.ALeft, fg=(1, 1, 0.4, 1), bg=(0.05, 0.06, 0.09, 0.7),
            mayChange=True,
        )
        self._help = OnscreenText(
            parent=a2d, pos=(-aspect + 0.05, -0.82), scale=0.032,
            align=TextNode.ALeft, fg=(0.8, 0.85, 0.9, 0.9), bg=(0.05, 0.06, 0.09, 0.55),
            mayChange=False,
            text=(
                "N/P test signs   ]/[ catalogue   PgUp/PgDn +-25   +/- speed\n"
                "R restart   V skeleton   ESC quit   |   search: type + Enter"
            ),
        )
        self._search = DirectEntry(
            parent=a2d, pos=(-aspect + 0.05, 0, -0.93), scale=0.045,
            width=14, numLines=1, initialText="", focus=0,
            command=self._on_search_enter,
            focusInCommand=self._on_search_focus, focusInExtraArgs=[True],
            focusOutCommand=self._on_search_focus, focusOutExtraArgs=[False],
            frameColor=(0.12, 0.14, 0.18, 0.9), text_fg=(1, 1, 1, 1),
        )
        # Catalogue neighbourhood list (clickable rows), upper-right.
        self._row_buttons: list[DirectButton] = []
        for i in range(LIST_ROWS):
            b = DirectButton(
                parent=a2d, pos=(aspect - 0.62, 0, 0.72 - i * 0.062),
                scale=0.038, frameColor=(0.1, 0.12, 0.16, 0.75),
                text="", text_align=TextNode.ALeft, text_fg=(0.9, 0.92, 0.95, 1),
                relief=1, pressEffect=1, command=self._on_row_click, extraArgs=[i],
                frameSize=(-0.3, 15.0, -0.55, 0.95),
            )
            self._row_buttons.append(b)

    def _on_search_focus(self, focused: bool) -> None:
        self._search_focused = focused

    def _on_search_enter(self, text: str) -> None:
        self._jump_prefix(text)
        self._search["focus"] = 0
        self._search_focused = False

    def _row_start(self) -> int:
        half = LIST_ROWS // 2
        start = self.cat_index - half
        return max(0, min(start, len(self.catalogue) - LIST_ROWS))

    def _on_row_click(self, row: int) -> None:
        idx = self._row_start() + row
        if 0 <= idx < len(self.catalogue):
            self._load_index(idx)

    def _refresh_ui(self) -> None:
        name, _ = self.catalogue[self.cat_index]
        in_test = self.cat_index in self.test_indices
        test_tag = (
            f"TEST {self.test_indices.index(self.cat_index) + 1}/{len(self.test_indices)}"
            if in_test else "catalogue"
        )
        self._status.setText(
            f"{name}   [{self.cat_index + 1}/{len(self.catalogue)} | {test_tag}]"
            f"   speed x{PLAYBACK_RATES[self.rate_i]:g}"
        )
        start = self._row_start()
        for i, b in enumerate(self._row_buttons):
            idx = start + i
            if idx >= len(self.catalogue):
                b["text"] = ""
                continue
            n = self.catalogue[idx][0]
            marker = "> " if idx == self.cat_index else ("* " if idx in self.test_indices else "  ")
            b["text"] = f"{marker}{n[:38]}"
            b["text_fg"] = (1, 1, 0.4, 1) if idx == self.cat_index else (0.9, 0.92, 0.95, 1)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        def guarded(fn, *args):
            def _inner():
                if not self._search_focused:
                    fn(*args)
            return _inner

        b = self.base
        b.accept("escape", b.userExit)
        b.accept("n", guarded(self._next_test, 1))
        b.accept("p", guarded(self._next_test, -1))
        b.accept("]", guarded(self._step_catalogue, 1))
        b.accept("[", guarded(self._step_catalogue, -1))
        b.accept("page_up", guarded(self._step_catalogue, -25))
        b.accept("page_down", guarded(self._step_catalogue, 25))
        b.accept("+", guarded(self._change_rate, 1))
        b.accept("=", guarded(self._change_rate, 1))   # unshifted +
        b.accept("-", guarded(self._change_rate, -1))
        b.accept("r", guarded(self._restart))

    def run(self) -> None:
        self.base.run()


def main() -> None:
    DebugViewer().run()


if __name__ == "__main__":
    main()
