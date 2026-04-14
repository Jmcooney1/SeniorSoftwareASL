import os
import sys
import subprocess

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QHBoxLayout,
    QSizePolicy, QComboBox,
)
from PySide6.QtCore import Qt

from david_module.sign_player import SignPlayerWidget


def get_tab() -> QWidget:
    return DavidModuleTab()


def _available_signs():
    """Return list of (name, Path) from the animation module."""
    try:
        from david_module.panda_port.animation import list_csv_signs
        return list_csv_signs()
    except Exception:
        return []


class DavidModuleTab(QWidget):
    """Wrapper tab that embeds the Panda3D sign viewer.

    The embedded player uses Panda3D's ``parent-window`` PRC variable to
    render directly inside the Qt widget.  If that fails, it automatically
    falls back to a popout subprocess.  A manual popout button is also
    available for debugging.
    """

    def __init__(self):
        super().__init__()
        self.proc = None
        self._signs: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        header = QLabel("David Module — Panda3D")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b;")
        layout.addWidget(header)

        btn_row = QHBoxLayout()
        sign_label = QLabel("Sign:")
        sign_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.sign_combo = QComboBox()
        self.sign_combo.setMinimumWidth(240)
        self.sign_combo.setMaxVisibleItems(20)
        self.sign_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._signs = _available_signs()
        self.sign_combo.addItems([name for name, _ in self._signs])
        if self._signs:
            self.sign_combo.setCurrentIndex(0)
        btn_row.addWidget(sign_label)
        btn_row.addWidget(self.sign_combo, 1)
        self.embed_btn = QPushButton("▶  Start Embedded")
        self.embed_btn.clicked.connect(self.start_embedded)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self.stop_embedded)
        self.launch_btn = QPushButton("Open Popout")
        self.launch_btn.clicked.connect(self.open_popout)
        btn_row.addWidget(self.embed_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.launch_btn)
        layout.addLayout(btn_row)

        self.status = QLabel("")
        layout.addWidget(self.status)

        # Reusable embedded player
        self.player = SignPlayerWidget()
        layout.addWidget(self.player, 1)

    def _selected_csv_path(self):
        idx = self.sign_combo.currentIndex()
        if 0 <= idx < len(self._signs):
            return str(self._signs[idx][1])
        return None

    def open_popout(self):
        if self.proc is not None and getattr(self.proc, "poll", lambda: 1)() is None:
            QMessageBox.information(self, "Already Running", "Panda process already running.")
            return

        csv_path = self._selected_csv_path()
        if csv_path is None:
            QMessageBox.warning(self, "No Sign", "No sign selected.")
            return

        script = os.path.join(os.path.dirname(__file__), "panda_port", "run_panda.py")
        try:
            self.proc = subprocess.Popen(
                [sys.executable, script, csv_path],
                cwd=os.path.dirname(script),
            )
            self.status.setText(f"Panda launched (popout) — {self.sign_combo.currentText()}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch Panda: {e}")

    def start_embedded(self):
        csv_path = self._selected_csv_path()
        if csv_path is None:
            QMessageBox.warning(self, "No Sign", "No sign selected.")
            return

        self.player.play(csv_path)
        if self.player.is_running:
            self.status.setText(f"Embedded — {self.sign_combo.currentText()}")
        else:
            self.status.setText("Embedded start failed — see player area.")

    def stop_embedded(self):
        self.player.stop()
        self.status.setText("Stopped.")

    def closeEvent(self, event):
        try:
            if self.proc is not None:
                self.proc.terminate()
        except Exception:
            pass
        self.player.stop()
        return super().closeEvent(event)
