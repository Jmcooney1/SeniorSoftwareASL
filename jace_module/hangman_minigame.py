import os
import sys
import random

from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QSizePolicy, QFrame,
    QStackedWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class HangmanTab(QWidget):
    def __init__(self):
        super().__init__()

        self.word_list = ["BIRD", "CAT", "HELLO", "SILLY GOOSE"]
        self.max_attempts = 10

        self.current_word = ""
        self.guessed_letters = set()
        self.attempts_left = self.max_attempts
        self.letter_buttons = {}

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setSpacing(20)

        self.word_label = QLabel("Word: ")
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setStyleSheet("font-size: 24px;")
        main_layout.addWidget(self.word_label)

        self.attempts_label = QLabel(f"Attempts left: {self.attempts_left}")
        self.attempts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.attempts_label)

        letters_layout = QHBoxLayout()
        letters_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(letters_layout)

        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            button = QPushButton(letter)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda checked, l=letter: self.guess_letter(l))
            letters_layout.addWidget(button)
            self.letter_buttons[letter] = button

        self.reset_game()

    def reset_game(self):
        self.current_word = random.choice(self.word_list)
        self.guessed_letters = set()
        self.attempts_left = self.max_attempts

        for button in self.letter_buttons.values():
            button.setEnabled(True)

        self.update_word_display()
        self.attempts_label.setText(f"Attempts left: {self.attempts_left}")

    def update_word_display(self):
        displayed = ""
        for char in self.current_word:
            if char == " ":
                displayed += "  "
            elif char in self.guessed_letters:
                displayed += char + " "
            else:
                displayed += "_ "
        self.word_label.setText(f"Word: {displayed.strip()}")

    def guess_letter(self, letter):
        if letter in self.guessed_letters:
            return

        self.guessed_letters.add(letter)
        self.letter_buttons[letter].setEnabled(False)

        if letter not in self.current_word:
            self.attempts_left -= 1

        self.update_word_display()
        self.attempts_label.setText(f"Attempts left: {self.attempts_left}")

        if self.is_word_guessed():
            QMessageBox.information(self, "You Win!", f"You guessed: {self.current_word}")
            self.reset_game()
        elif self.attempts_left <= 0:
            QMessageBox.information(self, "Game Over", f"The word was: {self.current_word}")
            self.reset_game()

    def is_word_guessed(self):
        for char in self.current_word:
            if char != " " and char not in self.guessed_letters:
                return False
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HangmanTab()
    window.resize(700, 600)
    window.show()
    sys.exit(app.exec())