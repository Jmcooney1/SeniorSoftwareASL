"""Entry point used by the root launcher to embed the sign-quiz tab."""

from PySide6.QtWidgets import QWidget

from jace_module_quiz.model_quiz_input import QuizInputWidget


def get_tab() -> QWidget:
    """Called by ``launcher.ModuleView`` — returns this module's tab content."""
    return QuizInputWidget()
