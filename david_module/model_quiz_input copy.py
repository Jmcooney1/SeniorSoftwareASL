"""Sign-quiz tab.

Shows a sign on the embedded Panda3D player and asks the user to type
what it is.  Built on top of ``david_module.sign_player.SignPlayerWidget``,
so it cooperates with the single-instance rule for the embedded player —
Panda is only spun up when the user presses *Play Sign*, and is torn down
when the user navigates away from this tab.
"""

import os
import random
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from david_module.sign_player import SignPlayerWidget


def _available_signs():
    """Return list of (display_name, csv_path) tuples from david_module."""
    try:
        from david_module.panda_port.animation import list_csv_signs
        return list_csv_signs()
    except Exception:
        return []


class QuizInputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quiz Input")

        self._signs = _available_signs()
        self._current_sign_name: str | None = None

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # Embedded Panda3D player.  Constructed but NOT started — Panda only
        # boots once the user presses Play Sign (or Next Sign).
        self.player = SignPlayerWidget()
        self.player.setMinimumHeight(260)
        main_layout.addWidget(self.player, 1)

        button_row = QHBoxLayout()
        self.play_button = QPushButton("▶  Play Sign")
        self.next_button = QPushButton("⏭  Next Sign")
        self.stop_button = QPushButton("■  Stop")
        self.play_button.clicked.connect(self.play_current_sign)
        self.next_button.clicked.connect(self.pick_next_sign)
        self.stop_button.clicked.connect(self.stop_player)
        button_row.addWidget(self.play_button)
        button_row.addWidget(self.next_button)
        button_row.addWidget(self.stop_button)
        main_layout.addLayout(button_row)

        # Question label
        self.question_label = QLabel("Press ▶ Play Sign to see a sign, then type its name below.")
        self.question_label.setStyleSheet("font-size: 16px;")
        self.question_label.setWordWrap(True)
        main_layout.addWidget(self.question_label)

        # Answer input box
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("Type your answer here")
        self.answer_input.returnPressed.connect(self.check_answer)
        main_layout.addWidget(self.answer_input)

        # Optional submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.check_answer)
        main_layout.addWidget(self.submit_button)

        # Feedback label for right/wrong messages
        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.feedback_label)

        if not self._signs:
            self.question_label.setText(
                "No sign CSVs found — check config.json's csv_dir setting."
            )
            self.play_button.setEnabled(False)
            self.next_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Player helpers
    # ------------------------------------------------------------------
    def _pick_random_sign(self) -> tuple[str, str] | None:
        if not self._signs:
            return None
        name, path = random.choice(self._signs)
        return name, str(path)

    def play_current_sign(self):
        if self._current_sign_name is None:
            self.pick_next_sign()
            return
        # Replay the same sign (hot-swap to the same CSV is a no-op cost).
        for name, path in self._signs:
            if name == self._current_sign_name:
                self.player.play(str(path))
                return

    def pick_next_sign(self):
        choice = self._pick_random_sign()
        if choice is None:
            return
        name, path = choice
        self._current_sign_name = name
        self.answer_input.clear()
        self.feedback_label.setText("")
        self.question_label.setText("What sign is this?")
        # play() will automatically tear down any other embedded player
        # elsewhere in the app before booting Panda3D for this widget.
        self.player.play(path)

    def stop_player(self):
        self.player.stop()

    def check_answer(self):
        if self._current_sign_name is None:
            self.feedback_label.setText("Press ▶ Play Sign first.")
            self.feedback_label.setStyleSheet("color: #b45309;")
            return

        guess = self.answer_input.text().strip().lower()
        target = self._current_sign_name.replace("_", " ").lower()
        if guess == target:
            self.feedback_label.setText(f"✅ Correct — it was '{self._current_sign_name}'.")
            self.feedback_label.setStyleSheet("color: #16a34a;")
        else:
            self.feedback_label.setText(
                f"❌ Not quite — answer was '{self._current_sign_name}'."
            )
            self.feedback_label.setStyleSheet("color: #dc2626;")

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizInputWidget()
    window.resize(600, 700)
    window.show()
    sys.exit(app.exec())
