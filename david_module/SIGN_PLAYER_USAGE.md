# Using SignPlayerWidget

Embed the Panda3D sign-language animation viewer in your module's tab.

## Quick Start

```python
from david_module.sign_player import SignPlayerWidget

# 1. Create the widget (add it to your layout wherever you want the 3D view)
player = SignPlayerWidget()
your_layout.addWidget(player, 1)   # stretch=1 so it fills available space

# 2. Play a sign (absolute path to a CSV file)
player.play("path/to/sign.csv")

# 3. Switch to a different sign (no restart needed — just call play again)
player.play("path/to/another_sign.csv")

# 4. Stop and tear down (frees resources; you can call play() again later)
player.stop()
```

## Finding Available Signs

Signs are loaded from the folder set by `csv_dir` in `config.json`
(default: `dataSet/david_dataset/best` — best-looking animations subset; the full
~4k-sign capture set lives in `dataSet/david_dataset/Landmarks`).

```python
from david_module.panda_port.animation import list_csv_signs, find_csv_sign

# List all available signs — returns list of (name, Path)
signs = list_csv_signs()
for name, csv_path in signs:
    print(name, csv_path)

# Look up a specific sign by name (case-insensitive)
path = find_csv_sign("Cat")   # returns Path or None
```

## API Reference

| Method / Property | Description |
|---|---|
| `player.play(csv_path)` | Start a sign or hot-swap to a new one |
| `player.stop()` | Tear down the 3D scene (replayable) |
| `player.is_running` | `True` if the viewer is active |
| `list_csv_signs()` | All signs as `(name, Path)` from `csv_dir` |
| `find_csv_sign(name)` | Look up one sign by name → `Path` or `None` |

## Example: Button Triggers Animation

```python
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from david_module.sign_player import SignPlayerWidget
from david_module.panda_port.animation import list_csv_signs


class MyTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.player = SignPlayerWidget()
        layout.addWidget(self.player, 1)

        self.signs = list_csv_signs()
        btn = QPushButton(f"Play: {self.signs[0][0]}")
        btn.clicked.connect(lambda: self.player.play(str(self.signs[0][1])))
        layout.addWidget(btn)
```

## Example: Play a Sign by Name

```python
from david_module.panda_port.animation import find_csv_sign

path = find_csv_sign("Cat")
if path:
    self.player.play(str(path))
```

## Notes

- The widget auto-resizes with your layout.
- If embedding fails on a platform, it falls back to a popout window automatically.
- Call `stop()` in your tab's `closeEvent` for clean shutdown.
- Only one `SignPlayerWidget` should be active at a time per process.
