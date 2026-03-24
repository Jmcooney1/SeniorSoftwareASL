import os
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# DATA_DIR is no longer hardcoded here — it's passed in by main() from drews_module/main.py
# This fallback only fires if you run translationquiz.py directly (not via the launcher)
_FALLBACK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataSet", "drew-dataset", "asl_letters"
)

IMG_SIZE      = (320, 320)
NUM_QUESTIONS = None  # None = infinite


def normalize(s: str) -> str:
    return s.strip().lower()


def load_questions(folder: str):
    """Return list of dicts: {"answer": "A", "path": ".../A.png"}"""
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Missing folder: {folder}\n"
            f"Create it and add images like A.png, B.png, ..."
        )

    questions = []
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            continue
        if len(base) != 1 or not base.isalpha():
            continue
        questions.append({"answer": base.upper(), "path": os.path.join(folder, name)})

    if not questions:
        raise RuntimeError(
            f"No valid images found in {folder}.\n"
            "Expected files like A.png, B.png, ..."
        )

    random.shuffle(questions)
    return questions


class ASLQuizApp:
    def __init__(self, root: tk.Tk, questions):
        self.root = root
        self.root.title("ASL Quiz")

        self.all_questions = questions[:]
        self.remaining     = questions[:]

        self.total_asked  = 0
        self.correct      = 0
        self.current      = None
        self.current_photo = None  # keep PhotoImage reference alive

        # ── UI ──
        self.image_label = tk.Label(root)
        self.image_label.pack(padx=12, pady=12)

        self.status_var = tk.StringVar(value="Score: 0/0")
        tk.Label(root, textvariable=self.status_var, font=("Arial", 12)).pack(pady=(0, 6))

        self.feedback_var = tk.StringVar(value="Type the letter and press Enter.")
        tk.Label(root, textvariable=self.feedback_var, font=("Arial", 11)).pack(pady=(0, 10))

        entry_frame = tk.Frame(root)
        entry_frame.pack(pady=(0, 12))

        tk.Label(entry_frame, text="Your answer:").pack(side=tk.LEFT, padx=(0, 8))
        self.answer_entry = tk.Entry(entry_frame, width=10, font=("Arial", 12))
        self.answer_entry.pack(side=tk.LEFT)
        self.answer_entry.bind("<Return>", self.on_submit)

        tk.Button(entry_frame, text="Submit", command=self.on_submit).pack(side=tk.LEFT, padx=8)
        tk.Button(root, text="Quit", command=root.quit).pack(pady=(0, 12))

        self.next_question()

    def update_status(self):
        self.status_var.set(f"Score: {self.correct}/{self.total_asked}")

    def load_and_show_image(self, path: str):
        img = Image.open(path).convert("RGBA")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.current_photo)

    def next_question(self):
        if not self.remaining:
            self.remaining = self.all_questions[:]
            random.shuffle(self.remaining)

        if NUM_QUESTIONS is not None and self.total_asked >= NUM_QUESTIONS:
            messagebox.showinfo("Done", f"Final score: {self.correct}/{self.total_asked}")
            self.root.quit()
            return

        self.current = self.remaining.pop()
        self.load_and_show_image(self.current["path"])
        self.feedback_var.set("What letter is this sign?")
        self.answer_entry.delete(0, tk.END)
        self.answer_entry.focus_set()

    def on_submit(self, event=None):
        if not self.current:
            return
        user = normalize(self.answer_entry.get())
        if user == "":
            return

        user_letter    = user[0].upper()
        correct_letter = self.current["answer"].upper()

        self.total_asked += 1
        if user_letter == correct_letter:
            self.correct += 1
            self.feedback_var.set(f"Correct! It was {correct_letter}.")
        else:
            self.feedback_var.set(
                f"Incorrect. It was {correct_letter}, you typed {user_letter}.")

        self.update_status()
        self.root.after(700, self.next_question)


def main(data_dir: str = None):
    """
    data_dir: resolved path to asl_letters folder.
    If None (e.g. running this file directly), falls back to _FALLBACK_DIR.
    """
    folder    = data_dir or _FALLBACK_DIR
    questions = load_questions(folder)
    root      = tk.Tk()
    ASLQuizApp(root, questions)
    root.mainloop()


if __name__ == "__main__":
    main()