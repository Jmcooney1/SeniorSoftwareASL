import importlib
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from . import predictor
from . import trainer

def get_tab() -> QWidget:
    # 1. Keep your Refresh Engine - it's necessary for code changes
    importlib.reload(predictor)
    importlib.reload(trainer)

    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)

    tabs = QTabWidget()
    
    # 2. Add your original cool styling here (Optional, but keeps it looking good)
    tabs.setStyleSheet("QTabBar::tab { padding: 12px; font-weight: bold; }")

    # 3. Instantiate the widgets
    pred_widget = predictor.PredictorWidget()
    train_widget = trainer.TrainerWidget()

    tabs.addTab(pred_widget, "🤖  Live Predictor")
    tabs.addTab(train_widget, "🎯  Motion Trainer")

    # 4. THE FIX: Logic to handle data refresh and camera safety
    def handle_tab_change(index):
        if index == 0:
            # Moving to Predictor: 
            # FIRST: Kill Trainer camera so Predictor can have it
            if hasattr(train_widget, '_stop_camera'): 
                train_widget._stop_camera()
            
            # SECOND: Force Predictor to re-read the .npy file right now
            # This is the "magic" line that makes the refresh work
            if hasattr(pred_widget, 'library'):
                pred_widget.library = pred_widget._load_library()
                if hasattr(pred_widget, 'status_bar'):
                    pred_widget.status_bar.setText(f"Library: {len(pred_widget.library)} gestures")
                    
        else:
            # Moving to Trainer: Kill Predictor camera
            if hasattr(pred_widget, '_stop_cam'): 
                pred_widget._stop_cam()

    tabs.currentChanged.connect(handle_tab_change)
    
    layout.addWidget(tabs)
    return wrapper