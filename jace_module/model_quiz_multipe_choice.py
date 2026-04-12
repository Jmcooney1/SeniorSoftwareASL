import sys

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QFrame, QPushButton, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt


class QuizInputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quiz Input - Multiple Choice")
        self.setGeometry(100, 100, 500, 500)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # --- 3D MODEL PLACEHOLDER ---
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

        # --- QUESTION BOX ---
        self.question_box = QFrame()
        self.question_box.setStyleSheet("""
            QFrame {
                border: 2px solid black;
                background-color: white;
            }
        """)

        question_layout = QVBoxLayout(self.question_box)

        self.question_label = QLabel("Question: What sign is this?")
        self.question_label.setStyleSheet("font-size: 16px;")
        question_layout.addWidget(self.question_label)

        # --- MULTIPLE CHOICE OPTIONS ---
        self.button_group = QButtonGroup(self)

        self.option_a = QRadioButton("A")
        self.option_b = QRadioButton("B")
        self.option_c = QRadioButton("C")
        self.option_d = QRadioButton("D")

        self.button_group.addButton(self.option_a)
        self.button_group.addButton(self.option_b)
        self.button_group.addButton(self.option_c)
        self.button_group.addButton(self.option_d)

        question_layout.addWidget(self.option_a)
        question_layout.addWidget(self.option_b)
        question_layout.addWidget(self.option_c)
        question_layout.addWidget(self.option_d)

        main_layout.addWidget(self.question_box)

        # --- SUBMIT BUTTON ---
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.check_answer)
        main_layout.addWidget(self.submit_button)

        self.setLayout(main_layout)

    def check_answer(self):
        selected_button = self.button_group.checkedButton()

        if selected_button:
            answer = selected_button.text()
            print("Selected answer:", answer)
        else:
            print("No option selected")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizInputWidget()
    window.show()
    sys.exit(app.exec())