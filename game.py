"""Game rules shared by self-play, search, evaluation, and the desktop interface.

The board is a flat array so a move can be represented by one integer:
    0 1 2
    3 4 5
    6 7 8
Cell values are +1 for X, -1 for O, and 0 for an empty square. Keeping the
rules here gives every part of the project the same definition of a game.
"""

import copy  # Search needs independent copies of a game to try possible moves.

import numpy as np  # Store the board and change its perspective with array arithmetic.


class TicTacToe:
    """Store a board and the player whose turn is next, without any UI logic."""

    def __init__(self):
        """Create an empty board with X moving first."""
        # int8 is enough for -1, 0, and +1. The network converts these integers
        # to floating-point numbers later because its weights are floats.
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1

    def reset(self):
        """Start another game using the same board array and return that array."""
        # fill changes the existing array in place. The desktop app can reuse
        # its game object instead of rebuilding its model and window.
        self.board.fill(0)
        self.current_player = 1
        return self.board

    def legal_moves(self):
        """Return the indices of empty cells, in board order."""
        # The caller checks whether the game is over before asking for a move;
        # this function only checks which cells are unoccupied.
        return [index for index in range(9) if self.board[index] == 0]

    def step(self, action):
        """Place the current player's mark at action, then switch turns.

        Callers must supply a legal move in an unfinished game. This small
        rules class does not validate actions or stop play automatically.
        """
        self.board[action] = self.current_player
        # Negating +1 gives -1 and vice versa. This also happens after a
        # winning move, which matters when MCTS evaluates a terminal board.
        self.current_player = -self.current_player

    def get_canonical(self):
        """Return a new board array expressed from the player-to-move viewpoint."""
        # Multiplication makes the mover's pieces +1 and the opponent's -1.
        # One network can therefore learn both sides with the same meaning
        # for its inputs. This changes signs, not board rotation or reflection.
        return self.board * self.current_player

    def check_winner(self):
        """Return +1 for an X win, -1 for O, 0 for a draw, or None if unfinished."""
        # These are the three rows, three columns, and two diagonals. Because
        # marks are +/-1, a sum of +/-3 means all three belong to one player.
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for line in lines:
            total = sum(self.board[index] for index in line)
            if total == 3:
                return 1
            if total == -3:
                return -1
        # Check wins before draws: the final empty cell might complete a line.
        if not self.legal_moves():
            return 0
        # None is different from 0: no result yet versus a completed draw.
        return None

    def is_over(self):
        """Report whether either player has won or the board is a draw."""
        # A truthiness check would miss draws because their result is zero.
        return self.check_winner() is not None

    def clone(self):
        """Copy the full game so simulated moves cannot alter the real board."""
        # A shallow copy would share the NumPy board array. deepcopy gives
        # each MCTS simulation and heuristic trial its own mutable board.
        return copy.deepcopy(self)
