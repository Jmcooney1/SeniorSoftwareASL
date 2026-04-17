"""Entry point used by the root launcher to embed the sign-quiz tab."""

from PySide6.QtWidgets import QWidget

from jace_module_hangman.hangman_minigame import HangmanGame


def get_tab() -> QWidget:
    """Called by ``launcher.ModuleView`` — returns this module's tab content."""
    return HangmanGame()