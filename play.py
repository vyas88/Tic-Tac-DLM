"""Play against the saved network through a small Tkinter desktop window.

The human is X (+1), moves first, and clicks one of nine board buttons. The
computer is O (-1) and chooses moves with the same MCTS implementation used
in training. This file loads weights for inference; it does not train them.
"""

import os  # Check that the checkpoint exists before building the game interface.
import tkinter as tk  # Window, labels, buttons, and the event loop.
from tkinter import messagebox  # Display an actionable error if weights are missing.

import numpy as np  # Select the most-visited MCTS move with argmax.
import torch  # Read the saved PyTorch parameter dictionary.

from game import TicTacToe  # Keep UI behavior consistent with the training game rules.
from mcts import MCTS  # Search beyond the network's initial move preferences.
from network import PolicyValueNet  # Rebuild the architecture before loading its weights.

MODEL_FILE = "tictactoe_model.pt"  # Resolved relative to the current working directory.
MCTS_SIMS = 40  # Search budget per computer move, balancing strength and response time.


class TicTacToeApp:
    """Connect one game state and one loaded model to Tkinter event callbacks."""

    def __init__(self, root):
        """Create the status label, nine move buttons, and a reset button."""
        self.root = root
        self.root.title("Play Tic-Tac-Toe")
        self.game = TicTacToe()
        self.net = self.load_model()  # Load once and reuse across moves and new games.
        self.status = tk.Label(root, text="Your turn: X", font=("Arial", 14))
        self.status.grid(row=0, column=0, columnspan=3, pady=8)
        self.buttons = []  # List index matches the corresponding board action.
        for action in range(9):
            # command expects a function to run later, not the result of calling
            # it now. The default argument move=action captures this iteration's
            # index; without it, every lambda would see the loop's final action.
            button = tk.Button(
                root, text="", width=5, height=2, font=("Arial", 24),
                command=lambda move=action: self.play_move(move),
            )
            # Integer division finds the board row, modulo finds the column.
            # Add one to the row because row zero holds the status label.
            button.grid(row=action // 3 + 1, column=action % 3, padx=2, pady=2)
            self.buttons.append(button)
        # Pass the bound reset method so Tkinter invokes it when clicked.
        tk.Button(root, text="New game", command=self.reset).grid(
            row=4, column=0, columnspan=3, pady=8
        )

    def load_model(self):
        """Build the default 64-unit network and load its saved parameters on CPU."""
        # A state_dict contains parameter values, so the layer names and shapes
        # must match training. If train.py's HIDDEN changes, this must match it.
        net = PolicyValueNet()
        # map_location allows CPU loading regardless of the saved device.
        # weights_only uses PyTorch's restricted loader for the weight dictionary.
        net.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
        # Select inference mode. MCTS calls predict(), which also disables
        # gradient tracking; loading/evaluation here do not update any weights.
        net.eval()
        return net

    def play_move(self, action):
        """Handle an enabled board button click, then schedule the computer's turn."""
        # Ignore clicks during O's turn or after completion. Occupied cells are
        # disabled by update_board(), so normal button clicks supply legal moves.
        if self.game.current_player != 1 or self.game.is_over():
            return
        self.game.step(action)
        self.update_board()
        if not self.finish_if_over():
            # after schedules a callback rather than blocking with sleep, giving
            # Tkinter time to redraw the human move first. It does not create a
            # background thread: the later search still runs on the UI thread.
            self.root.after(200, self.computer_move)

    def computer_move(self):
        """Search the current board, apply the most-visited move, and redraw."""
        # Root noise is a self-play training aid. Disable it for the playable
        # opponent and choose the largest visit probability instead of sampling.
        policy = MCTS(self.net, n_simulations=MCTS_SIMS).run(
            self.game, add_noise=False
        )
        self.game.step(int(np.argmax(policy)))
        self.update_board()
        self.finish_if_over()

    def update_board(self):
        """Render marks and disable occupied cells or the entire finished board."""
        # The model uses numeric marks; the interface translates them to symbols.
        symbols = {1: "X", -1: "O", 0: ""}
        for action, button in enumerate(self.buttons):
            button.config(text=symbols[int(self.game.board[action])])
            if self.game.board[action] != 0 or self.game.is_over():
                button.config(state="disabled")
            else:
                button.config(state="normal")
        # Preserve the result message set by finish_if_over() for a finished game.
        if not self.game.is_over():
            self.status.config(text="Your turn: X")

    def finish_if_over(self):
        """Show the result and return True if play has ended, otherwise False."""
        winner = self.game.check_winner()
        # None means ongoing; zero means a finished draw and must be handled.
        if winner is None:
            return False
        if winner == 0:
            message = "Draw."
        elif winner == 1:
            message = "You win!"
        else:
            message = "Computer wins."
        self.status.config(text=message)
        # Refresh once more so no board button stays enabled after the result.
        self.update_board()
        return True

    def reset(self):
        """Clear the game and refresh the existing buttons, keeping the loaded model."""
        self.game.reset()
        self.update_board()


def main():
    """Create the application window and run Tkinter's event loop."""
    root = tk.Tk()
    if not os.path.exists(MODEL_FILE):
        # Give a clear next step instead of letting torch.load fail for a
        # missing file. The window is destroyed because no game can start yet.
        messagebox.showerror("Model missing", "Run python train.py before opening the game.")
        root.destroy()
        return
    TicTacToeApp(root)
    # mainloop keeps the window alive and dispatches clicks and after callbacks.
    # Without it the script would finish before the user could interact.
    root.mainloop()


# Importing the class should not create a desktop window as a side effect.
if __name__ == "__main__":
    main()
