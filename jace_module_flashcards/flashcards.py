import os
import sys
import random

from PySide6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QSizePolicy, QFrame,
    QStackedWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataSet",
    "drew_dataset",
    "asl_letters"
)

IMG_SIZE = (256, 256)

class FlashcardsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.current_image_path = None
        self.current_definition = ""

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.setSpacing(20)

        self.card_frame = QFrame()
        self.card_frame.setFixedSize(420, 420)
        self.card_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 3px solid black;
                border-radius: 12px;
            }
        """)

        card_layout = QVBoxLayout(self.card_frame)

        self.card_stack = QStackedWidget()
        card_layout.addWidget(self.card_stack)

        # Front side
        self.front_widget = QWidget()
        front_layout = QVBoxLayout(self.front_widget)
        front_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.front_hint = QLabel("Front: Image")
        self.front_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        front_layout.addWidget(self.image_label)

        # Back side
        self.back_widget = QWidget()
        back_layout = QVBoxLayout(self.back_widget)
        back_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.definition_label = QLabel("Definition goes here")
        self.definition_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.definition_label.setWordWrap(True)
        self.definition_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            font-family: Arial, sans-serif;
            color: #000000;
        """)

        self.back_hint = QLabel("Back: Definition")
        self.back_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        back_layout.addWidget(self.definition_label)

        self.card_stack.addWidget(self.front_widget)  # 0
        self.card_stack.addWidget(self.back_widget)   # 1

        button_layout = QHBoxLayout()

        self.flip_button = QPushButton("Show Definition")
        self.flip_button.clicked.connect(self.flip_card)

        self.next_button = QPushButton("Next Flashcard")
        self.next_button.clicked.connect(self.load_random_image)

        button_layout.addWidget(self.flip_button)
        button_layout.addWidget(self.next_button)

        main_layout.addWidget(self.card_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(button_layout)

        self.load_random_image()

    def load_random_image(self):
        all_images = []

        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_images.append(os.path.join(root, file))

        if not all_images:
            QMessageBox.warning(self, "No Images", f"No images found in:\n{DATASET_PATH}")
            self.image_label.setText("No images found.")
            self.definition_label.setText("No definition available.")
            return

        self.current_image_path = random.choice(all_images)

        pixmap = QPixmap(self.current_image_path)
        if pixmap.isNull():
            self.image_label.setText("Failed to load image.")
            self.definition_label.setText("Failed to load definition.")
            return

        scaled = pixmap.scaled(
            *IMG_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

        self.current_definition = os.path.splitext(os.path.basename(self.current_image_path))[0]
        self.definition_label.setText(self.current_definition)

        self.card_stack.setCurrentIndex(0)
        self.flip_button.setText("Show Definition")

    def flip_card(self):
        if self.card_stack.currentIndex() == 0:
            self.card_stack.setCurrentIndex(1)
            self.flip_button.setText("Show Image")
        else:
            self.card_stack.setCurrentIndex(0)
            self.flip_button.setText("Show Definition")


class FlashCardWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        label = QLabel("Welcome to the Jace Module!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        inner_tabs = QTabWidget()
        inner_tabs.addTab(FlashcardsTab(), "Flashcards")
        layout.addWidget(inner_tabs)


def get_tab() -> QWidget:
    return FlashCardWidget()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlashCardWidget()
    window.resize(700, 600)
    window.show()
    sys.exit(app.exec())