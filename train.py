import csv
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from baselines import heuristic_agent, random_agent
from game import TicTacToe
from mcts import MCTS
from network import PolicyValueNet

SEED = 42
GENERATIONS = 20
GAMES_PER_GEN = 30
MCTS_SIMS = 40
EVAL_SIMS = 25
C_PUCT = 1.5
HIDDEN = 64
LR = 1e-3
BATCH_SIZE = 64
TRAIN_EPOCHS = 5
BUFFER_GENERATIONS = 5
EVAL_GAMES = 50
TEMP_MOVES = 3


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def self_play_game(net):
    game = TicTacToe()
    examples = []
    move_number = 0
    while not game.is_over():
        policy = MCTS(net, C_PUCT, MCTS_SIMS).run(game)
        board = game.get_canonical().copy()
        player = game.current_player
        if move_number < TEMP_MOVES:
            action = np.random.choice(9, p=policy)
        else:
            action = int(np.argmax(policy))
        examples.append((board, policy, player))
        game.step(action)
        move_number += 1

    winner = game.check_winner()
    samples = []
    for board, policy, player in examples:
        target = 0.0 if winner == 0 else float(1 if winner == player else -1)
        samples.append((board, policy, target))
    return samples


def train_network(net, optimizer, buffer):
    losses = []
    for _ in range(TRAIN_EPOCHS):
        net.train()
        random.shuffle(buffer)
        for start in range(0, len(buffer), BATCH_SIZE):
            batch = buffer[start:start + BATCH_SIZE]
            boards = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32)
            policies = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.float32)
            values = torch.tensor([item[2] for item in batch], dtype=torch.float32)

            logits, predicted_values = net(boards)
            # This teaches the value head the final outcome from this position.
            value_loss = F.mse_loss(predicted_values.squeeze(1), values)
            # This teaches the policy head to copy MCTS's improved visit policy.
            policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss = value_loss + policy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def play_match(net, opponent_fn, agent_plays_first):
    game = TicTacToe()
    agent_player = 1 if agent_plays_first else -1
    while not game.is_over():
        if game.current_player == agent_player:
            policy = MCTS(net, C_PUCT, EVAL_SIMS).run(game, add_noise=False)
            action = int(np.argmax(policy))
        else:
            action = opponent_fn(game)
        game.step(action)

    winner = game.check_winner()
    if winner == 0:
        return 0
    return 1 if winner == agent_player else -1


def evaluate(net, opponent_fn):
    results = [play_match(net, opponent_fn, game_index % 2 == 0)
               for game_index in range(EVAL_GAMES)]
    # Alternating X and O removes first-move advantage bias.
    wins = results.count(1)
    draws = results.count(0)
    losses = results.count(-1)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / EVAL_GAMES,
    }


def save_plots(random_scores, heuristic_scores):
    generations = range(1, GENERATIONS + 1)
    for scores, title, filename in [
        (random_scores, "Score vs Random Agent", "winrate_vs_random.png"),
        (heuristic_scores, "Score vs Heuristic Agent", "winrate_vs_heuristic.png"),
    ]:
        plt.figure()
        plt.plot(generations, scores)
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title(title)
        plt.ylim(0, 1)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    plt.figure()
    plt.plot(generations, random_scores, label="Random")
    plt.plot(generations, heuristic_scores, label="Heuristic")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("AlphaZero Tic-Tac-Toe Scores")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("winrate_combined.png", bbox_inches="tight")
    plt.close()


def main():
    seed_everything()
    net = PolicyValueNet(HIDDEN)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    generation_buffer = deque(maxlen=BUFFER_GENERATIONS)
    random_scores = []
    heuristic_scores = []
    metrics = []

    for generation in range(1, GENERATIONS + 1):
        generation_data = []
        for _ in range(GAMES_PER_GEN):
            generation_data.extend(self_play_game(net))
        generation_buffer.append(generation_data)
        samples = [sample for data in generation_buffer for sample in data]
        loss = train_network(net, optimizer, samples)
        random_result = evaluate(net, random_agent)
        heuristic_result = evaluate(net, heuristic_agent)
        random_scores.append(random_result["score"])
        heuristic_scores.append(heuristic_result["score"])
        metrics.append([generation, loss, random_result["score"], heuristic_result["score"]])
        print(
            f"Gen {generation:02d} | loss {loss:.3f} | "
            f"random {random_result['score']:.2f} "
            f"({random_result['wins']}/{random_result['draws']}/{random_result['losses']}) | "
            f"heuristic {heuristic_result['score']:.2f} "
            f"({heuristic_result['wins']}/{heuristic_result['draws']}/{heuristic_result['losses']})"
        )

    save_plots(random_scores, heuristic_scores)
    with open("metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "loss", "score_vs_random", "score_vs_heuristic"])
        writer.writerows(metrics)


if __name__ == "__main__":
    main()
