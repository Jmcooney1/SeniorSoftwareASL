from panda3d.core import loadPrcFileData

# Ensure no top-level window is created for this test
loadPrcFileData("", "window-type none\n")

import importlib
pd = importlib.import_module("panda_main")

app = pd.PandaApp()
print("PandaApp created; anim folder used:", getattr(app, 'landmark_animator', None) is not None)
if getattr(app, 'landmark_animator', None) is not None:
    print("Animator enabled:", app.landmark_animator.enabled)
    try:
        print("Anim dir:", app.landmark_animator.anim_dir)
    except Exception:
        pass
