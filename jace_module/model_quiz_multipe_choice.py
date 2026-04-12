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

        self.answers = {
            "A": "Cat",
            "B": "Dog",
            "C": "Bird",
            "D": "Fish"
        }

        self.correct_answer = "Cat"

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

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

        self.question_box = QFrame()
        self.question_box.setStyleSheet("""
            QFrame {
                border: 2px solid black;
                background-color: white;
            }
            QRadioButton {
                color: black;
                font-size: 15px;
                padding: 4px;
            }
            QLabel {
                color: black;
                font-size: 16px;
            }
        """)

        question_layout = QVBoxLayout(self.question_box)

        self.question_label = QLabel("Question: What sign is this?")
        question_layout.addWidget(self.question_label)

        self.button_group = QButtonGroup(self)

        self.option_a = QRadioButton(f"A. {self.answers['A']}")
        self.option_b = QRadioButton(f"B. {self.answers['B']}")
        self.option_c = QRadioButton(f"C. {self.answers['C']}")
        self.option_d = QRadioButton(f"D. {self.answers['D']}")

        self.option_a.setProperty("answer", self.answers["A"])
        self.option_b.setProperty("answer", self.answers["B"])
        self.option_c.setProperty("answer", self.answers["C"])
        self.option_d.setProperty("answer", self.answers["D"])

        self.button_group.addButton(self.option_a)
        self.button_group.addButton(self.option_b)
        self.button_group.addButton(self.option_c)
        self.button_group.addButton(self.option_d)

        question_layout.addWidget(self.option_a)
        question_layout.addWidget(self.option_b)
        question_layout.addWidget(self.option_c)
        question_layout.addWidget(self.option_d)

        main_layout.addWidget(self.question_box)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.check_answer)
        main_layout.addWidget(self.submit_button)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    def check_answer(self):
        selected_button = self.button_group.checkedButton()

        if not selected_button:
            self.result_label.setText("No option selected")
            return

        answer = selected_button.property("answer")

        if answer == self.correct_answer:
            self.result_label.setText("Correct!")
        else:
            self.result_label.setText("Wrong!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizInputWidget()
    window.show()
    sys.exit(app.exec())