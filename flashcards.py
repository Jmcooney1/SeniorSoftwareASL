import tkinter as tk

cards = [
    {"front": "Word A", "back": "Image A"},
    {"front": "Word B", "back": "Image B"},
    {"front": "Word C", "back": "Image C"},
]

current_card = 0
showing_front = True


def update_card():
    if showing_front:
        card_text.config(text=cards[current_card]["front"])
    else:
        card_text.config(text=cards[current_card]["back"])
    progress_label.config(text=f"Card {current_card + 1} / {len(cards)}")

def previous_card():
    global current_card, showing_front
    current_card = (current_card - 1) % len(cards)
    showing_front = True
    update_card()



def flip_card():
    global showing_front
    showing_front = not showing_front
    update_card()


def next_card():
    global current_card, showing_front
    current_card = (current_card + 1) % len(cards)
    showing_front = True
    update_card()

def flip_key(event):
    flip_card()

def next_key(event):
    next_card()

def previous_key(event):
    previous_card()


root = tk.Tk()
root.title("ASL Flashcards")
root.geometry("500x400")

top_frame = tk.Frame(root)
top_frame.pack(pady=10)

flip_button = tk.Button(top_frame, text="Flip [Space]", command=flip_card)
flip_button.pack()

card_text = tk.Label(root, text="", font=("Arial", 24), width=20, height=6, relief="solid")
card_text.pack(pady=30)

next_button = tk.Button(root, text="Next [->]", command=next_card)
next_button.pack(side="right", padx=20)

previous_button = tk.Button(root, text="[<-] Previous", command=previous_card)
previous_button.pack(side="left", padx=20)

progress_label = tk.Label(root, text="", font=("Arial", 12))
progress_label.pack(side="bottom", pady=20)

root.bind("<Left>", previous_key)
root.bind("<Right>", next_key)
root.bind("<space>", flip_key)

update_card()
root.mainloop()