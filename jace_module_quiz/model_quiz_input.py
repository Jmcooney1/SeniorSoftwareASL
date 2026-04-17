import os
import sys
import random

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
        self.setGeometry(100, 100, 500, 500)
        
        self._signs = _available_signs()
        self._current_sign_name: str | None = None

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        #Model player box
        self.player = SignPlayerWidget()
        self.player.setMinimumHeight(260)
        main_layout.addWidget(self.player, 1)
        
        button_row = QHBoxLayout()
        self.play_button = QPushButton("▶  Play Sign")
        self.next_button = QPushButton("⏭  Next Sign")
        self.stop_button = QPushButton("■  Stop")
        self.play_button.clicked.connect(self.play_current_sign)
        self.next_button.clicked.connect(self.pick_next_sign)
        self.stop_button.clicked.connect(self.player.stop)
        button_row.addWidget(self.play_button)
        button_row.addWidget(self.next_button)
        button_row.addWidget(self.stop_button)
        main_layout.addLayout(button_row)


        # Question label
        self.question_label = QLabel("Question: What sign is this?")
        self.question_label.setStyleSheet("font-size: 16px;")
        main_layout.addWidget(self.question_label)

        # Answer input box
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("Type your answer here")
        main_layout.addWidget(self.answer_input)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.check_answer)
        main_layout.addWidget(self.submit_button)

        self.setLayout(main_layout)

    def check_answer(self):
        answer = self.answer_input.text()
        print("User answer:", answer)

    

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
        #self.feedback_label.setText("")
        self.question_label.setText("What sign is this?")
        # play() will automatically tear down any other embedded player
        # elsewhere in the app before booting Panda3D for this widget.
        self.player.play(path)

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizInputWidget()
    window.show()
    sys.exit(app.exec())