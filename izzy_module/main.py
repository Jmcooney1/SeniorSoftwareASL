"""
google_module/main.py
Exposes get_tab() -> QWidget for the root launcher.
Contains two sub-tabs: Live Predictor and Motion Trainer.
"""
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_tab() -> QWidget:
    """Called by the root launcher — returns this module's tab content."""
    from izzy_module.predictor import PredictorWidget
    from izzy_module.trainer   import TrainerWidget

    wrapper = QWidget()
    layout  = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    tabs = QTabWidget()
    tabs.setStyleSheet("""
        QTabWidget::pane {
            border: none;
            background: #0f172a;
        }
        QTabBar::tab {
            background: #1e293b;
            color: #64748b;
            padding: 8px 20px;
            font-size: 13px;
            font-weight: bold;
            border: none;
            border-bottom: 2px solid transparent;
        }
        QTabBar::tab:selected {
            background: #0f172a;
            color: #38bdf8;
            border-bottom: 2px solid #38bdf8;
        }
        QTabBar::tab:hover:!selected {
            background: #334155;
            color: #cbd5e1;
        }
    """)

    tabs.addTab(PredictorWidget(), "🤖  Live Predictor")
    tabs.addTab(TrainerWidget(),   "🎯  Motion Trainer")

    layout.addWidget(tabs)
    return wrapper