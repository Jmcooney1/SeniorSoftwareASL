import os
import sys
import subprocess
import importlib

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QHBoxLayout, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt


def get_tab() -> QWidget:
    return DavidModuleTab()


class DavidModuleTab(QWidget):
    """A simple wrapper tab that can launch the Panda app.

    Embedding is experimental — it will try to parent the Panda window
    into this widget, and falls back to launching a separate popout
    process if embedding is not supported on the platform/Panda build.
    """

    def __init__(self):
        super().__init__()
        self.proc = None
        self.panda_app = None
        self.timer = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        header = QLabel("David Module — Panda3D")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b;")
        layout.addWidget(header)

        btn_row = QHBoxLayout()
        self.launch_btn = QPushButton("Open Panda (Popout)")
        self.launch_btn.clicked.connect(self.open_popout)
        self.embed_btn = QPushButton("Start Embedded (Slow)")
        self.embed_btn.clicked.connect(self.start_embedded)
        btn_row.addWidget(self.launch_btn)
        btn_row.addWidget(self.embed_btn)
        layout.addLayout(btn_row)

        self.status = QLabel("")
        layout.addWidget(self.status)

        # Container for Panda widget
        self.embed_container = QWidget()
        self.embed_container.setMinimumHeight(720)
        self.embed_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._embed_layout = QVBoxLayout(self.embed_container)
        self._embed_layout.setContentsMargins(0, 0, 0, 0)
        # give the embed container stretch so it takes available space
        layout.addWidget(self.embed_container, 1)

    def open_popout(self):
        if self.proc is not None and getattr(self.proc, "poll", lambda: 1)() is None:
            QMessageBox.information(self, "Already Running", "Panda process already running.")
            return

        script = os.path.join(os.path.dirname(__file__), "panda_port", "run_panda.py")
        try:
            self.proc = subprocess.Popen([sys.executable, script], cwd=os.path.dirname(script))
            self.status.setText("Panda launched (popout).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch Panda: {e}")

    def start_embedded(self):
        self.status.setText("Starting embedded (QPanda3D)...")
        pkg_dir = os.path.join(os.path.dirname(__file__), "panda_port")
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

        try:
            import traceback, subprocess, sys as _sys
            from QPanda3D.QPanda3DWidget import QPanda3DWidget
        except Exception as exc:
            # Gather diagnostic information to help resolve environment issues
            import traceback as _tb
            tb = _tb.format_exc()
            try:
                pip_check = subprocess.run([_sys.executable, "-m", "pip", "show", "QPanda3D"], capture_output=True, text=True, timeout=5)
                pip_out = pip_check.stdout.strip() or pip_check.stderr.strip()
            except Exception as pip_exc:
                pip_out = f"pip check failed: {pip_exc}"

            msg = (
                f"Failed to import QPanda3D: {exc}\n\n"
                f"Python executable: {_sys.executable}\n"
                f"pip show QPanda3D output:\n{pip_out}\n\n"
                f"Traceback:\n{tb}"
            )
            QMessageBox.critical(self, "QPanda3D Import Error", msg)
            self.status.setText("QPanda3D import failed — see dialog for details.")
            return

        try:
            import qpanda_adapter
        except Exception as e:
            QMessageBox.critical(self, "Adapter Error", f"Failed to import embed adapter: {e}")
            return

        try:
            # Remove any existing embedded widget
            if getattr(self, "panda_widget", None) is not None:
                try:
                    self._embed_layout.removeWidget(self.panda_widget)
                    self.panda_widget.deleteLater()
                except Exception:
                    pass

            # Create the QPanda3D world and widget
            world = qpanda_adapter.QPandaPandaWorld(width=self.embed_container.width() or 1024, height=self.embed_container.height() or 768)
            # QPanda3D expects the actual Panda3DWorld instance; adapter stores it on _world
            real_world = getattr(world, "_world", world)
            panda_widget = QPanda3DWidget(real_world)
            self._embed_layout.addWidget(panda_widget)
            self.panda_widget = panda_widget
            self.status.setText("Embedded Panda started.")
            self.embed_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Embedded start failed", str(e))

    def _step_panda(self):
        if self.panda_app is not None:
            try:
                # Advance Panda's global clock so task.time advances correctly
                try:
                    from panda3d.core import ClockObject
                    ClockObject.getGlobalClock().tick()
                except Exception:
                    pass

                # Step the Panda task manager and request a render
                try:
                    self.panda_app.taskMgr.step()
                except Exception:
                    pass

                try:
                    # Render a frame so the visuals update when embedding
                    if hasattr(self.panda_app, "graphicsEngine"):
                        try:
                            self.panda_app.graphicsEngine.renderFrame()
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                # Ignore stepping errors — embedding is experimental
                pass

    def closeEvent(self, event):
        # Stop any running popout process
        try:
            if self.proc is not None:
                self.proc.terminate()
        except Exception:
            pass
        try:
            if self.timer is not None:
                self.timer.stop()
        except Exception:
            pass
        return super().closeEvent(event)
