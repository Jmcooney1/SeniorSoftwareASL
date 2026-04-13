from panda3d.core import loadPrcFileData

# Ensure no top-level window is created for this test
loadPrcFileData("", "window-type none\n")

import importlib
pd = importlib.import_module("panda_main")

app = pd.PandaApp()
print("PandaApp created; animator present:", getattr(app, 'landmark_animator', None) is not None)
if getattr(app, 'landmark_animator', None) is not None:
    print("Backend:", getattr(app.landmark_animator, "backend", "<unknown>"))
    print("Animator enabled:", app.landmark_animator.enabled)
    try:
        print("Selected clip:", app.landmark_animator.selected_clip_path)
    except Exception:
        pass
    try:
        print("Selected source:", app.landmark_animator.selected_clip_source)
    except Exception:
        pass
