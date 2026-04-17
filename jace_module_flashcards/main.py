"""Entry point used by the root launcher to embed the sign-quiz tab."""

from PySide6.QtWidgets import QWidget

from jace_module_flashcards.flashcards import FlashcardsGame


def get_tab() -> QWidget:
    """Called by ``launcher.ModuleView`` — returns this module's tab content."""
    return FlashcardsGame()