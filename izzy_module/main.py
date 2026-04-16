import importlib
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

def get_tab() -> QWidget:
    # 1. Manually reload the sub-scripts to force them to re-read the .npy file
    from . import predictor
    from . import trainer
    importlib.reload(predictor)
    importlib.reload(trainer)

    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    
    tabs = QTabWidget()
    # ... (keep your existing stylesheet here) ...

    # 2. Instantiate the freshly reloaded widgets
    pred_widget = predictor.PredictorWidget()
    train_widget = trainer.TrainerWidget()

    tabs.addTab(pred_widget, "🤖  Live Predictor")
    tabs.addTab(train_widget, "🎯  Motion Trainer")

    # Camera cleanup logic
    def handle_tab_change(index):
        if index == 0:
            if hasattr(train_widget, '_stop_camera'): train_widget._stop_camera()
        else:
            if hasattr(pred_widget, '_stop_cam'): pred_widget._stop_cam()

    tabs.currentChanged.connect(handle_tab_change)
    layout.addWidget(tabs)
    return wrapper