import os
import sys
import random
import numpy as np
import subprocess


from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt,QTimer
from PyQt6.QtGui import QPixmap



base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
os.chdir(os.path.join(parent_dir, 'googleMedaPipe'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
asl_library = np.load(os.path.join(parent_dir, 'googleMedaPipe', 'asl_library.npy'), allow_pickle=True).item()


class HangmanTab(QWidget):
    BRIDGE_FILE = '/tmp/asl_detected_letter.txt'
    def accuracy_test(self):
        
        if os.path.exists(self.BRIDGE_FILE):
            os.remove(self.BRIDGE_FILE)
        
        bridge_path = os.path.join(base_dir, 'accuracy_test_bridge.py')  # ← base_dir points to jace_module
        self.asl_process = subprocess.Popen(
        [sys.executable, bridge_path],
        cwd=os.path.join(parent_dir, 'googleMedaPipe')  # ← still run it FROM googleMedaPipe so it finds asl_library.npy
        )

        self.asl_timer = QTimer()
        self.asl_timer.timeout.connect(self.check_asl_letter)
        self.asl_timer.start(500)
    def __init__(self):
        super().__init__()

        self.word_list = ["BIRD", "CAT", "HELLO", "SILLY GOOSE"]
        self.max_attempts = 8  # Updated to match 8 stages

        self.current_word = ""
        self.guessed_letters = set()
        self.attempts_left = self.max_attempts
        self.letter_buttons = {}

        # Load all 9 hangman images (stage 0–8)
        self.hangman_images = []
        for i in range(9):
            path = os.path.join(os.path.dirname(__file__), "hangman_png", f"hangman_{i}.png")
            self.hangman_images.append(QPixmap(path))

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setSpacing(20)

        # Hangman image display
        self.hangman_label = QLabel()
        self.hangman_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.hangman_label)

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

        button = QPushButton("Static Letter Accuracy Test")
        buttonMotion = QPushButton("Motion Accuracy Test")
        button.clicked.connect(self.accuracy_test)
       # buttonMotion.clicked.connect(self.accuracy_test)
        main_layout.addWidget(button)
        main_layout.addWidget(buttonMotion)

        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            button = QPushButton(letter)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda checked, l=letter: self.guess_letter(l))
            letters_layout.addWidget(button)
            self.letter_buttons[letter] = button

        self.reset_game()

    def update_hangman_image(self):
        # Stage = how many wrong guesses have been made
        stage = self.max_attempts - self.attempts_left
        stage = max(0, min(stage, 8))  # Clamp between 0 and 8
        self.hangman_label.setPixmap(
            self.hangman_images[stage].scaled(
                300, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def reset_game(self):
        self.current_word = random.choice(self.word_list)
        self.guessed_letters = set()
        self.attempts_left = self.max_attempts

        for button in self.letter_buttons.values():
            button.setEnabled(True)

        self.update_word_display()
        self.update_hangman_image()
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
            self.update_hangman_image()  # Only update image on wrong guess

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

    def check_asl_letter(self):
        if os.path.exists(self.BRIDGE_FILE):
            with open(self.BRIDGE_FILE, 'r') as f:
                letter = f.read().strip()
            if letter:
                os.remove(self.BRIDGE_FILE)  # clear it so same letter isn't read twice
                self.guess_letter(letter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HangmanTab()
    window.resize(700, 600)
    window.show()
    sys.exit(app.exec())