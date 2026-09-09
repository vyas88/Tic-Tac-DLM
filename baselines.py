import random


def random_agent(game):
    return random.choice(game.legal_moves())


def heuristic_agent(game):
    legal_moves = game.legal_moves()
    for player in (game.current_player, -game.current_player):
        for action in legal_moves:
            trial = game.clone()
            trial.current_player = player
            trial.step(action)
            if trial.check_winner() == player:
                return action

    if 4 in legal_moves:
        return 4
    for action in (0, 2, 6, 8):
        if action in legal_moves:
            return action
    return legal_moves[0]
