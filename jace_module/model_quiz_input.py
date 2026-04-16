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

import sys

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QFrame, QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt



class QuizInputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quiz Input")
        self.setGeometry(100, 100, 500, 500)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Placeholder box for 3D model
        self.model_box = QFrame()
        self.model_box.setFixedHeight(220)
        self.model_box.setStyleSheet("""
            QFrame {
                border: 2px solid black;
                background-color: lightgray;
            }
        """)

        model_layout = QVBoxLayout(self.model_box)
        model_label = QLabel("3D Model Placeholder")
        model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_layout.addWidget(model_label)

        main_layout.addWidget(self.model_box)

        # Question label
        self.question_label = QLabel("Question: What sign is this?")
        self.question_label.setStyleSheet("font-size: 16px;")
        main_layout.addWidget(self.question_label)

        # Answer input box
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("Type your answer here")
        main_layout.addWidget(self.answer_input)

        # Optional submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.check_answer)
        main_layout.addWidget(self.submit_button)

        self.setLayout(main_layout)

    def check_answer(self):
        answer = self.answer_input.text()
        print("User answer:", answer)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizInputWidget()
    window.show()
    sys.exit(app.exec())