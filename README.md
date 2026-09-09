# Mini-AlphaZero for Tic-Tac-Toe

This is a small teaching implementation of the AlphaZero recipe for Tic-Tac-Toe. A policy-value neural network suggests promising moves and estimates who is winning, while Monte Carlo Tree Search (MCTS) uses those predictions to choose stronger moves. The network learns from self-play games that were improved by MCTS.

## How to run

```bash
pip install -r requirements.txt
```

Train a new model and generate the charts:

```bash
python train.py
```

Play against the trained agent in a small desktop window:

```bash
python play.py
```

## File map

- `game.py`: Tic-Tac-Toe board rules and game state.
- `network.py`: Small PyTorch policy-value neural network.
- `mcts.py`: Network-guided Monte Carlo Tree Search.
- `baselines.py`: Random and rule-based opponent agents.
- `train.py`: Self-play, training, evaluation, plots, and CSV output.
- `play.py`: Small Tkinter board for playing against the trained agent.
- `requirements.txt`: Required Python packages.

`tictactoe_model.pt` is the saved PyTorch model, so it is correctly shown as a binary file by VS Code. You play as X and the trained MCTS agent plays as O. On macOS, Tkinter may print a deprecation warning when the game opens; it is harmless and does not affect play.

## Expected result

The score against the random agent should climb toward about 0.95 to 1.0. The score against the heuristic agent should climb toward about 0.5 and then plateau.

## VIVA notes

### Why does the network have two heads (policy and value)?

The policy head suggests which moves are worth searching. The value head estimates the eventual result, so MCTS can judge a position without playing every game to the end.

### What does MCTS add on top of the raw network?

The raw network makes one prediction. MCTS repeatedly explores likely moves, compares their results, and turns the network prediction into a better move policy.

### Why no random rollouts (unlike classic MCTS)?

AlphaZero uses the learned value head to evaluate leaf positions. This is usually more informed and less noisy than finishing games with random moves.

### Why flip the value sign during backpropagation?

The value is always from the player-to-move viewpoint. Moving one level up changes whose viewpoint is being considered, so a good result for one player is bad for the other.

### Why use visit counts as the training target for the policy, not the raw network policy?

Visit counts represent the policy after MCTS has searched and improved the network's initial guess. Training on them teaches the network to imitate stronger search decisions.

### What is the canonical board and why use it?

It is the board multiplied by the current player, so the mover's pieces are always `+1`. This lets one network learn one consistent perspective instead of learning separate X and O views.

### Why does the score vs the heuristic bot plateau near 0.5 instead of reaching 1.0?

Tic-Tac-Toe is a forced draw under good play. A strong agent draws a good opponent instead of beating it, so a score near 0.5 means it is drawing every game and playing optimally.

### What is `c_puct` and what does raising or lowering it do?

`c_puct` controls the exploration bonus in MCTS. Raising it explores more moves, while lowering it focuses more strongly on moves that already look best.

### What is Dirichlet noise at the root for?

It slightly changes root move priors during self-play, encouraging varied openings and a more diverse training dataset. It is turned off during evaluation.
