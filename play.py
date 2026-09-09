import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import torch

from game import TicTacToe
from mcts import MCTS
from network import PolicyValueNet

MODEL_FILE = "tictactoe_model.pt"
MCTS_SIMS = 40


class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Play Tic-Tac-Toe")
        self.game = TicTacToe()
        self.net = self.load_model()
        self.status = tk.Label(root, text="Your turn: X", font=("Arial", 14))
        self.status.grid(row=0, column=0, columnspan=3, pady=8)
        self.buttons = []
        for action in range(9):
            button = tk.Button(
                root, text="", width=5, height=2, font=("Arial", 24),
                command=lambda move=action: self.play_move(move),
            )
            button.grid(row=action // 3 + 1, column=action % 3, padx=2, pady=2)
            self.buttons.append(button)
        tk.Button(root, text="New game", command=self.reset).grid(
            row=4, column=0, columnspan=3, pady=8
        )

    def load_model(self):
        net = PolicyValueNet()
        net.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
        net.eval()
        return net

    def play_move(self, action):
        if self.game.current_player != 1 or self.game.is_over():
            return
        self.game.step(action)
        self.update_board()
        if not self.finish_if_over():
            # A short delay lets the player see their move before the reply.
            self.root.after(200, self.computer_move)

    def computer_move(self):
        policy = MCTS(self.net, n_simulations=MCTS_SIMS).run(
            self.game, add_noise=False
        )
        self.game.step(int(np.argmax(policy)))
        self.update_board()
        self.finish_if_over()

    def update_board(self):
        symbols = {1: "X", -1: "O", 0: ""}
        for action, button in enumerate(self.buttons):
            button.config(text=symbols[int(self.game.board[action])])
            if self.game.board[action] != 0 or self.game.is_over():
                button.config(state="disabled")
            else:
                button.config(state="normal")
        if not self.game.is_over():
            self.status.config(text="Your turn: X")

    def finish_if_over(self):
        winner = self.game.check_winner()
        if winner is None:
            return False
        if winner == 0:
            message = "Draw."
        elif winner == 1:
            message = "You win!"
        else:
            message = "Computer wins."
        self.status.config(text=message)
        self.update_board()
        return True

    def reset(self):
        self.game.reset()
        self.update_board()


def main():
    root = tk.Tk()
    if not os.path.exists(MODEL_FILE):
        messagebox.showerror("Model missing", "Run python train.py before opening the game.")
        root.destroy()
        return
    TicTacToeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
