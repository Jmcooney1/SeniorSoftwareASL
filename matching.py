import tkinter as tk
import random
import os
from PIL import Image, ImageTk

# letters you currently have image files for
ALL_LETTERS = [
    "A", "B", "C", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]

CARDS_PER_GAME = 12
COLUMNS = 6
ROWS = 4
TILE_WIDTH = 120
TILE_HEIGHT = 100
IMAGE_SIZE = (80, 80)
MISMATCH_DELAY = 500

cards = []
card_images = {}
card_frames = []
card_labels = []

first_choice = None
second_choice = None
lock_board = False
matches_found = 0
attempts = 0


def build_cards():
    chosen_letters = random.sample(ALL_LETTERS, CARDS_PER_GAME)
    new_cards = []

    for i, letter in enumerate(chosen_letters, start=1):
        image_path = os.path.join("data", "asl_letters", f"{letter.lower()}.jpg")

        new_cards.append({
            "pair": i,
            "kind": "text",
            "value": letter,
            "matched": False,
            "revealed": False
        })

        new_cards.append({
            "pair": i,
            "kind": "image",
            "value": image_path,
            "matched": False,
            "revealed": False
        })

    random.shuffle(new_cards)
    return new_cards


def load_images():
    global card_images
    card_images = {}

    for card in cards:
        if card["kind"] == "image":
            path = card["value"]

            if not os.path.exists(path):
                print(f"Missing image: {path}")
                continue

            img = Image.open(path)
            img = img.resize(IMAGE_SIZE)
            card_images[path] = ImageTk.PhotoImage(img, master=root)


def update_progress():
    progress_label.config(text=f"Pairs matched: {matches_found} / {CARDS_PER_GAME}")

def update_attempts():
    attempts_label.config(text=f"Attempts: {attempts}")

def reveal_card(index):
    card = cards[index]

    if card["kind"] == "text":
        card_labels[index].config(text=card["value"], image="", fg="black")
    else:
        if card["value"] in card_images:
            card_labels[index].config(text="", image=card_images[card["value"]])
        else:
            card_labels[index].config(text="Missing\nImage", image="", fg="black")

    card_frames[index].config(relief="sunken")


def hide_card(index):
    card_labels[index].config(text="?", image="", bg="white", fg="black")
    card_frames[index].config(relief="raised", bg="white")


def mark_matched(index):
    card_frames[index].config(relief="sunken", bg="#d9ffd9")
    card_labels[index].config(bg="#d9ffd9", fg="black")


def on_card_click(index):
    global first_choice, second_choice, lock_board, attempts

    if lock_board:
        return
    if cards[index]["matched"]:
        return
    if cards[index]["revealed"]:
        return

    cards[index]["revealed"] = True
    reveal_card(index)

    if first_choice is None:
        first_choice = index
        return

    second_choice = index
    attempts += 1
    update_attempts()
    check_match()


def check_match():
    global first_choice, second_choice, lock_board, matches_found

    if first_choice is None or second_choice is None:
        return

    if cards[first_choice]["pair"] == cards[second_choice]["pair"]:
        cards[first_choice]["matched"] = True
        cards[second_choice]["matched"] = True

        mark_matched(first_choice)
        mark_matched(second_choice)

        matches_found += 1
        update_progress()

        first_choice = None
        second_choice = None
    else:
        lock_board = True
        root.after(MISMATCH_DELAY, hide_unmatched)


def hide_unmatched():
    global first_choice, second_choice, lock_board

    if first_choice is not None:
        cards[first_choice]["revealed"] = False
        hide_card(first_choice)

    if second_choice is not None:
        cards[second_choice]["revealed"] = False
        hide_card(second_choice)

    first_choice = None
    second_choice = None
    lock_board = False


def restart_game():
    global cards, first_choice, second_choice, lock_board, matches_found, attempts
    global card_frames, card_labels

    first_choice = None
    second_choice = None
    lock_board = False
    matches_found = 0
    attempts = 0

    for frame in card_frames:
        frame.destroy()

    card_frames = []
    card_labels = []

    cards = build_cards()
    load_images()
    create_board()
    update_progress()


def create_board():
    for i in range(len(cards)):
        frame = tk.Frame(
            board_frame,
            width=TILE_WIDTH,
            height=TILE_HEIGHT,
            bg="white",
            relief="raised",
            bd=2
        )
        frame.grid(row=i // COLUMNS, column=i % COLUMNS, padx=6, pady=6)
        frame.grid_propagate(False)

        label = tk.Label(
        frame,
        text="?",
        font=("Arial", 20),
        bg="white",
        fg="black"
        )
        label.place(relx=0.5, rely=0.5, anchor="center")

        frame.bind("<Button-1>", lambda event, i=i: on_card_click(i))
        label.bind("<Button-1>", lambda event, i=i: on_card_click(i))

        card_frames.append(frame)
        card_labels.append(label)


root = tk.Tk()
root.title("ASL Matching Game")
window_width = 950
window_height = 650
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(False, False)

title_label = tk.Label(root, text="ASL Matching Game", font=("Arial", 20))
title_label.place(relx=0.5, rely=0.06, anchor="center")

board_frame = tk.Frame(root)
board_frame.place(relx=0.5, rely=0.5, anchor="center")

progress_label = tk.Label(root, text="", font=("Arial", 14))
progress_label.place(relx=0.5, rely=0.90, anchor="center")

restart_button = tk.Button(root, text="New Game", font=("Arial", 12), command=restart_game)
restart_button.place(relx=0.5, rely=0.96, anchor="center")

attempts_label = tk.Label(root, text="", font=("Arial", 14))
attempts_label.place(relx=0.02, rely=0.94, anchor="w")

cards = build_cards()
load_images()
create_board()
update_progress()
update_attempts()
root.mainloop()