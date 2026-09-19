"""Train the network by repeatedly learning from its own MCTS-guided games.

The learning cycle is self-play -> training examples -> weight updates ->
evaluation. Each example stores (canonical board, search policy, final outcome).
The policy target comes from search; the value target comes from who actually
won. No human move labels or baseline-opponent moves are used as training data.

Running this file writes metrics.csv, three score plots, and the model weights
used by play.py. Importing its functions does not start the training run.
"""

import csv  # Save numeric results in a format usable by spreadsheets and plotting tools.
import random  # Shuffle training samples; baselines.py also uses this generator.
from collections import deque  # Bound replay history and automatically remove old data.

import matplotlib.pyplot as plt  # Save evaluation curves after the training run.
import numpy as np  # Board arrays, probability-based move sampling, and mean loss.
import torch  # Network tensors, gradients, optimizer, and saved weights.
import torch.nn.functional as F  # MSE and log_softmax for the two learning objectives.

from baselines import heuristic_agent, random_agent  # Fixed comparison opponents.
from game import TicTacToe  # Shared rules used in self-play and evaluation.
from mcts import MCTS  # Turn the current network's predictions into search targets.
from network import PolicyValueNet  # Learn move preferences and expected outcomes together.

# These are small teaching-run settings, not claimed optimal hyperparameters.
SEED = 42  # Starting point for repeatable random sequences in the same environment.
GENERATIONS = 20  # Each generation collects games, trains, and evaluates the new weights.
GAMES_PER_GEN = 30  # More games provide more positions but increase self-play time.
MCTS_SIMS = 40  # Search effort per self-play move; also affects the policy targets.
EVAL_SIMS = 25  # Smaller evaluation budget keeps repeated comparison matches quicker.
C_PUCT = 1.5  # Weigh prior-guided exploration against backed-up values in MCTS.
HIDDEN = 64  # Width of both hidden layers; must match the architecture loading the weights.
LR = 1e-3  # Adam learning rate (0.001), controlling the scale of weight updates.
BATCH_SIZE = 64  # Maximum examples per update; a final smaller batch is still used.
TRAIN_EPOCHS = 5  # Revisit all retained examples five times per generation.
BUFFER_GENERATIONS = 5  # Mix recent experience while discarding increasingly old targets.
EVAL_GAMES = 50  # Games per baseline; an even count gives equal turns as X and O.
TEMP_MOVES = 3  # Sample the first three individual moves, then choose maximum visits.
MODEL_FILE = "tictactoe_model.pt"  # Binary weight checkpoint also expected by play.py.


def seed_everything():
    """Seed each random-number generator used by this CPU training script."""
    # These libraries have separate generators. Python handles shuffling and
    # random baseline moves; NumPy handles move sampling and Dirichlet noise;
    # PyTorch initializes the network's weights. Seeding one does not seed all.
    # Seeds aid reproducibility but do not guarantee identical runs across
    # different library versions, hardware, or nondeterministic operations.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def self_play_game(net):
    """Play both sides with net and return (board, policy, outcome) examples."""
    game = TicTacToe()
    examples = []  # Keep the mover's identity until the eventual winner is known.
    move_number = 0
    while not game.is_over():
        # A fresh search tree uses the same network for either side. Root
        # noise is enabled by default to diversify the self-play experience.
        policy = MCTS(net, C_PUCT, MCTS_SIMS).run(game)
        # Save the state before the chosen move, because that is the position
        # to which this policy belongs. get_canonical already creates an array;
        # copy makes the intention to retain an independent snapshot explicit.
        board = game.get_canonical().copy()
        player = game.current_player
        # Sampling early moves explores different games. Later argmax choices
        # concentrate on the move search visited most. This is a simple switch
        # between sampling and greedy play, not a gradual temperature schedule.
        if move_number < TEMP_MOVES:
            action = np.random.choice(9, p=policy)  # Sample early moves for variety
        else:
            action = int(np.argmax(policy))  # Choose the most preferred move
        # Retain the full nine-entry policy, not just the move actually played:
        # it contains search's relative preference for alternative legal moves.
        examples.append((board, policy, player))
        game.step(action)
        move_number += 1

    # Only the finished game supplies a supervised value target. Earlier
    # network estimates guided search but are not substituted for this result.
    winner = game.check_winner()
    samples = []
    for board, policy, player in examples:
        # Convert the absolute X/O winner to the player whose board we stored:
        # win +1, draw 0, loss -1. Every position from that player's side gets
        # the same final result; there is no discounted reward in this game.
        target = 0.0 if winner == 0 else float(1 if winner == player else -1)
        samples.append((board, policy, target))
    return samples  # Training data from one complete game


def train_network(net, optimizer, buffer):
    """Train on a nonempty list of examples and return mean loss across batches.

    buffer is the flattened replay list, not the deque of generation lists.
    It is shuffled in place. The optimizer persists across generations so Adam
    keeps its running gradient statistics as the network continues learning.
    """
    losses = []
    for _ in range(TRAIN_EPOCHS):
        # predict() switches the model to evaluation mode during self-play.
        # Restore training mode here. This does not itself perform an update.
        net.train()  # Enable training mode
        # Mix positions from different games to reduce the effect of their
        # original chronological order on successive gradient updates.
        random.shuffle(buffer)  # Mix samples before each training pass
        for start in range(0, len(buffer), BATCH_SIZE):
            batch = buffer[start:start + BATCH_SIZE]  # Select one batch

            # A batch has B boards and policy targets shaped (B, 9), plus
            # outcome targets shaped (B,). float32 matches the network weights.
            # np.array stacks individual boards/policies before tensor creation.
            boards = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32)
            policies = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.float32)
            values = torch.tensor([item[2] for item in batch], dtype=torch.float32)

            logits, predicted_values = net(boards)
            # MSE penalizes distance from the observed -1/0/+1 result. squeeze(1)
            # changes (B, 1) to (B,), matching targets and preventing accidental
            # broadcasting across examples. It preserves the batch axis if B=1.
            value_loss = F.mse_loss(predicted_values.squeeze(1), values)
            # Cross-entropy with the full MCTS distribution is
            # -sum(target_probability * log(predicted_probability)). log_softmax
            # computes log probabilities stably. Sum over nine moves, then
            # average over boards. This teaches search preferences, not a
            # one-hot label for only the selected move.
            policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            # Equal weighting lets gradients from both heads update the shared
            # body. No separate loss weights or regularization term are added.
            loss = value_loss + policy_loss

            # PyTorch accumulates gradients unless cleared. Compute this batch's
            # derivatives, then let Adam use them to adjust registered parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # item stores a plain number, avoiding references to training graphs.
            losses.append(loss.item())
    # This is a mean of batch losses across epochs, not an evaluation accuracy.
    # Each batch has equal weight here, including any final smaller batch.
    return float(np.mean(losses))


def play_match(net, opponent_fn, agent_plays_first):
    """Return +1, 0, or -1 for an agent win, draw, or loss against one baseline."""
    game = TicTacToe()
    agent_player = 1 if agent_plays_first else -1  # AI plays X or O
    while not game.is_over():
        if game.current_player == agent_player:
            # Measure play with the evaluation search budget, without self-play
            # noise or sampled agent moves. A random opponent still adds randomness.
            policy = MCTS(net, C_PUCT, EVAL_SIMS).run(game, add_noise=False)
            action = int(np.argmax(policy))  # AI chooses its preferred move
        else:
            # Passing a function allows the same match code to evaluate either
            # baseline, since both accept a game and return a legal action.
            action = opponent_fn(game)
        game.step(action)

    # Convert X/O signs to the evaluated agent's perspective, even when it is O.
    winner = game.check_winner()
    if winner == 0:
        return 0  # Draw
    return 1 if winner == agent_player else -1  # AI win or loss


def evaluate(net, opponent_fn):
    """Count match results and return a score with half credit for draws."""
    # Alternate starting positions to balance first-move advantage. With the
    # configured even EVAL_GAMES, the agent plays each side equally often.
    results = [play_match(net, opponent_fn, game_index % 2 == 0)
               for game_index in range(EVAL_GAMES)]
    wins = results.count(1)
    draws = results.count(0)
    losses = results.count(-1)
    # Score is not the fraction of games won. For example, 0.5 can mean all
    # draws or an equal number of wins and losses; inspect the counts to tell.
    # A score alone does not establish optimal play against every opponent.
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / EVAL_GAMES,  # Win = 1 point; draw = 0.5
    }


def save_plots(random_scores, heuristic_scores):
    """Save per-opponent and combined curves from one score per generation."""
    generations = range(1, GENERATIONS + 1)
    # The historical filenames say "winrate", but each y-value is the score
    # defined in evaluate(), including half a point for a draw.
    for scores, title, filename in [
        (random_scores, "Score vs Random Agent", "winrate_vs_random.png"),
        (heuristic_scores, "Score vs Heuristic Agent", "winrate_vs_heuristic.png"),
    ]:
        plt.figure()  # Start a separate figure so opponent curves do not overlap.
        plt.plot(generations, scores)
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title(title)
        plt.ylim(0, 1)  # A shared full score range makes the two plots comparable.
        plt.savefig(filename, bbox_inches="tight")  # Include labels with less empty margin.
        plt.close()  # Release each figure after saving instead of leaving it open.

    # Compare both opponents on one graph
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
    """Run the complete learning cycle and save results in the working directory."""
    seed_everything()
    # Start from fresh weights each run; this function does not resume the
    # existing checkpoint. play.py is the file that loads saved weights.
    net = PolicyValueNet(HIDDEN)  # Create an untrained network
    # Adam tracks gradient averages and squared-gradient averages to adapt
    # parameter updates. net.parameters() includes the body and both heads.
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    # Each deque item is an entire generation's examples. maxlen limits the
    # number of generations, not the number of boards: game lengths vary.
    generation_buffer = deque(maxlen=BUFFER_GENERATIONS)
    random_scores = []
    heuristic_scores = []
    metrics = []

    for generation in range(1, GENERATIONS + 1):
        # Collect all games with the current weights before training on them.
        generation_data = []
        for _ in range(GAMES_PER_GEN):
            generation_data.extend(self_play_game(net))  # Collect self-play examples

        # Combining recent generations reuses experience and mixes positions
        # from slightly different policies. Appending beyond maxlen drops the oldest.
        generation_buffer.append(generation_data)
        # Flatten into a new list so train_network can shuffle examples without
        # changing the order or membership of the deque's generation lists.
        samples = [sample for data in generation_buffer for sample in data]
        loss = train_network(net, optimizer, samples)  # Learn from self-play

        # Evaluate updated weights. These matches measure progress and do not
        # become training examples or contribute gradients to the network.
        random_result = evaluate(net, random_agent)  # Test against random moves
        heuristic_result = evaluate(net, heuristic_agent)  # Test against rule-based moves
        random_scores.append(random_result["score"])
        heuristic_scores.append(heuristic_result["score"])
        metrics.append([generation, loss, random_result["score"], heuristic_result["score"]])

        # Counts make the half-draw score interpretable. Loss measures fit to
        # replay targets and need not move in lockstep with playing strength.
        print(
            f"Gen {generation:02d} | loss {loss:.3f} | "
            f"random {random_result['score']:.2f} "
            f"({random_result['wins']}/{random_result['draws']}/{random_result['losses']}) | "
            f"heuristic {heuristic_result['score']:.2f} "
            f"({heuristic_result['wins']}/{heuristic_result['draws']}/{heuristic_result['losses']})"
        )

    # Filenames are relative to the process's working directory. A new run
    # replaces the previous plots, CSV, and checkpoint when it finishes.
    save_plots(random_scores, heuristic_scores)
    # newline="" lets the csv module manage line endings itself.
    with open("metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "loss", "score_vs_random", "score_vs_heuristic"])
        writer.writerows(metrics)  # Save results for every generation
    # state_dict stores named weights/biases, not the network's Python code or
    # the optimizer/replay history. play.py rebuilds a matching architecture
    # before loading these weights; this file is not a full training-resume state.
    torch.save(net.state_dict(), MODEL_FILE)


# The entry-point guard permits imports for small experiments or checks without
# unexpectedly starting all 20 generations of training.
if __name__ == "__main__":
    main()
