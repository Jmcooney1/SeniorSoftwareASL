"""
kily_module/dictionary_ui.py
Full translator UI — returned by main.get_tab().
"""
from __future__ import annotations

import os
import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget, QSplitter,
)

from kily_module.main import LANDMARK_FOLDER
from dataSet.david_dataset.csvPoses import CSVPoses
from david_module.sign_player import SignPlayerWidget


class TranslatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset: CSVPoses | None = None

        self._build_ui()
        self._apply_theme()
        QTimer.singleShot(100, self._load_dataset)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _load_dataset(self):
        """Load CSVPoses from the landmark folder supplied by main.py."""
        try:
            # Point at the world-pose subfolder — same place csv_animation.py uses
            world_pose_dir = os.path.join(LANDMARK_FOLDER, "world_pose")
            if not os.path.isdir(world_pose_dir):
                world_pose_dir = LANDMARK_FOLDER  # fallback to root landmark folder

            self._dataset = CSVPoses(dataset_path=world_pose_dir)
            self._dataset.build_dictionary()

            n = len(self._dataset.name_to_csv)
            self._db_status.setText(f"✅ {n} signs loaded")
            self._db_status.setStyleSheet("color: #4ade80; font-size: 12px;")
            self._run_btn.setEnabled(True)
            self._populate_table()
            self._log(f"Dataset ready — {n} signs available.")
        except Exception as exc:
            self._db_status.setText(f"❌ {exc}")
            self._db_status.setStyleSheet("color: #f87171; font-size: 12px;")
            self._log(f"Failed to load dataset: {exc}")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _play_word(self, word: str):
        """Look up the CSV for *word* and send it to the SignWidget."""
        if self._dataset is None:
            return
        try:
            csv_path = self._dataset.get_pose_csv(word)
        except ValueError:
            self._log(f"⚠️  '{word}' not found in dataset.")
            return

        self._now_playing.setText(f"Now playing: {word}")
        self._sign_widget.play(csv_path)
        self._log(f"▶  Playing: {word}")

    def _on_translate(self):
        """Play the first recognised word from the input field."""
        text = self._input.text().strip()
        if not text or self._dataset is None:
            return
        words = [w for w in re.split(r"[;,\s]+", text) if w]
        if words:
            self._play_word(words[0])

    def _on_table_clicked(self, row: int, _col: int):
        """Clicking a row in the database tab immediately plays that sign."""
        item = self._table.item(row, 0)
        if item:
            self._play_word(item.text())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Title + status
        title = QLabel("ASL Sign Viewer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 4px;")
        root.addWidget(title)

        self._db_status = QLabel("⏳ Loading dataset…")
        self._db_status.setStyleSheet("color: gray; font-size: 12px;")
        root.addWidget(self._db_status)

        # Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a word to sign…")
        self._input.returnPressed.connect(self._on_translate)
        self._input.textChanged.connect(self._on_input_changed)
        self._run_btn = QPushButton("▶  Play")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_translate)
        input_row.addWidget(self._input)
        input_row.addWidget(self._run_btn)
        root.addLayout(input_row)

        # Main splitter: left = viewer+database stacked, right = log
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left side: vertical splitter (60% viewer / 40% database) ────
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # TOP 60% — 3D viewer
        viewer_widget = QWidget()
        vt_layout = QVBoxLayout(viewer_widget)
        vt_layout.setContentsMargins(4, 4, 4, 4)

        self._now_playing = QLabel("Select a word to begin")
        self._now_playing.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._now_playing.setStyleSheet("font-size: 15px; font-weight: bold; padding: 4px;")
        vt_layout.addWidget(self._now_playing)

        self._sign_widget = SignPlayerWidget(parent=self)
        vt_layout.addWidget(self._sign_widget, stretch=1)

        left_splitter.addWidget(viewer_widget)

        # BOTTOM 40% — database browser
        db_widget = QWidget()
        dt_layout = QVBoxLayout(db_widget)
        dt_layout.setContentsMargins(4, 4, 4, 4)

        db_label = QLabel("📖  Database")
        db_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #38bdf8; padding: 2px 0;")
        dt_layout.addWidget(db_label)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search signs…")
        self._search.textChanged.connect(self._filter_table)
        dt_layout.addWidget(self._search)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Sign", "CSV File"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.cellClicked.connect(self._on_table_clicked)
        dt_layout.addWidget(self._table)

        left_splitter.addWidget(db_widget)

        # Set 60/40 split
        left_splitter.setStretchFactor(0, 6)
        left_splitter.setStretchFactor(1, 4)

        main_splitter.addWidget(left_splitter)

        # ── Right side: log ──────────────────────────────────────────────
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumWidth(280)
        main_splitter.addWidget(self._log_box)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        root.addWidget(main_splitter, stretch=1)

    def _populate_table(self):
        if self._dataset is None:
            return
        items = sorted(self._dataset.name_to_csv.items())
        self._table.setRowCount(len(items))
        for row, (name, path) in enumerate(items):
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem(os.path.basename(path)))

    def _filter_table(self, text: str):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            hide = text.lower() not in (item.text().lower() if item else "")
            self._table.setRowHidden(row, hide)

    def _on_input_changed(self, text: str):
        """Highlight matching rows in the database tab as the user types."""
        if self._dataset is None:
            return
        typed = {w.lower() for w in re.split(r"[;,\s]+", text.strip()) if w}
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            hit = item and item.text().lower() in typed
            colour = QColor("#fef08a") if hit else QColor("transparent")
            for col in range(self._table.columnCount()):
                cell = self._table.item(row, col)
                if cell:
                    cell.setBackground(colour)

    def _log(self, msg: str):
        self._log_box.append(msg)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self):
        self.setStyleSheet("""
            QWidget            { background-color: #0a0a0a; }
            QLabel             { color: #e2e8f0; }
            QLineEdit          { background-color: #1e293b; color: #f1f5f9;
                                 border: 1px solid #334155; border-radius: 6px;
                                 padding: 6px 10px; }
            QLineEdit:focus    { border-color: #2563eb; }
            QPushButton        { background-color: #2563eb; color: white;
                                 border: none; border-radius: 6px;
                                 padding: 7px 18px; font-weight: bold; }
            QPushButton:hover  { background-color: #1d4ed8; }
            QPushButton:pressed{ background-color: #1e40af; }
            QPushButton:disabled { background-color: #1e293b; color: #475569; }
            QTabWidget::pane   { border: 1px solid #1e293b; background-color: #0a0a0a; }
            QTabBar::tab       { background-color: #111827; color: #64748b;
                                 padding: 8px 20px; font-size: 13px; font-weight: bold;
                                 border: none; border-bottom: 2px solid transparent; }
            QTabBar::tab:selected      { background-color: #0a0a0a; color: #38bdf8;
                                         border-bottom: 2px solid #38bdf8; }
            QTabBar::tab:hover:!selected { background-color: #1e293b; color: #cbd5e1; }
            QTextEdit          { background-color: #050505; color: #a3e635;
                                 border: 1px solid #1e293b; border-radius: 4px;
                                 font-family: monospace; font-size: 12px; }
            QTableWidget       { background-color: #0f172a; color: #e2e8f0;
                                 gridline-color: #1e293b; border: none; }
            QTableWidget::item:selected { background-color: #1e3a5f; color: white; }
            QHeaderView::section { background-color: #1e293b; color: #94a3b8;
                                   padding: 6px; border: none; font-weight: bold; }
            QSplitter::handle  { background-color: #1e293b; width: 2px; }
        """)