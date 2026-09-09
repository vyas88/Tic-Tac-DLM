import copy

import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self.board

    def legal_moves(self):
        return [index for index in range(9) if self.board[index] == 0]

    def step(self, action):
        self.board[action] = self.current_player
        self.current_player = -self.current_player

    def get_canonical(self):
        # This is always from the mover's view, so one perspective is enough.
        return self.board * self.current_player

    def check_winner(self):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for line in lines:
            total = sum(self.board[index] for index in line)
            if total == 3:
                return 1
            if total == -3:
                return -1
        if not self.legal_moves():
            return 0
        return None

    def is_over(self):
        return self.check_winner() is not None

    def clone(self):
        return copy.deepcopy(self)
