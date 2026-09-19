"""Simple opponents used to measure the trained agent's playing strength.

Both functions accept an unfinished TicTacToe game and return a legal action
without changing that game. Neither baseline learns or uses the network.
"""

import random  # Python's seeded generator selects the random baseline's moves.


def random_agent(game):
    """Choose uniformly from empty cells to provide a basic comparison opponent."""
    # Sampling only legal moves avoids choosing an already occupied square.
    return random.choice(game.legal_moves())


def heuristic_agent(game):
    """Try to win, block a loss, then prefer center, a corner, and finally an edge.

    This is a one-move rule-based opponent, not a complete minimax solver. It
    does not explicitly detect forks or guarantee perfect Tic-Tac-Toe play.
    """
    legal_moves = game.legal_moves()
    # First test our own immediate wins, then the opponent's. This ordering
    # chooses a win now over blocking a threat that would matter next turn.
    for player in (game.current_player, -game.current_player):
        for action in legal_moves:
            # A clone keeps each hypothetical placement separate. Temporarily
            # setting current_player also lets us test the opponent's threats
            # using the same step() and check_winner() rules.
            trial = game.clone()
            trial.current_player = player
            trial.step(action)
            if trial.check_winner() == player:
                # On the opponent pass, taking this square ourselves blocks
                # that winning line. The real game's player stays unchanged.
                return action

    # Center participates in four winning lines; a corner participates in
    # three. These simple priorities help when no immediate tactic is found.
    if 4 in legal_moves:
        return 4
    # Fixed order makes the heuristic deterministic when several corners work.
    for action in (0, 2, 6, 8):
        if action in legal_moves:
            return action
    return legal_moves[0]  # Remaining empty cells are edges; choose the first.
