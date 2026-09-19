"""A small neural network that supplies both predictions needed by MCTS.

Input boards use the current player's perspective from game.get_canonical().
The policy head scores nine possible moves; the value head estimates the
final result for that same player. train.py teaches both heads from self-play.
"""

import numpy as np  # Return move probabilities in the array format MCTS uses.
import torch  # Tensors, gradient tracking, and tensor-to-number conversion.
import torch.nn as nn  # Layers and the base class that registers trainable weights.
import torch.nn.functional as F  # Stateless operations such as softmax.


class PolicyValueNet(nn.Module):
    """Share board features between a move policy and an outcome estimate."""

    def __init__(self, hidden_size=64):
        """Build two hidden layers and separate policy/value output layers."""
        # nn.Module's initialization lets PyTorch discover these layers for
        # optimizer updates and state_dict checkpoint saving/loading.
        super().__init__()
        # A 9-cell board is small enough for fully connected layers. Each
        # Linear learns weighted combinations; ReLU adds nonlinearity so the
        # network can represent patterns beyond a single linear transformation.
        # Sequential applies these layers in the order written below.
        self.body = nn.Sequential(
            nn.Linear(9, hidden_size), nn.ReLU(),  # Nine cells become hidden features.
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),  # Combine those features.
        )
        self.policy_head = nn.Linear(hidden_size, 9)  # One raw move score per cell.
        self.value_head = nn.Linear(hidden_size, 1)  # One outcome estimate per board.

    def forward(self, x):
        """Map a float batch (B, 9) to move logits (B, 9) and values (B, 1)."""
        # Both tasks update the same feature extractor during training, so
        # move selection and outcome prediction can learn useful board patterns.
        features = self.body(x)
        # Keep scores as logits here: train.py uses log_softmax directly for a
        # numerically stable policy loss. predict() converts them to probabilities.
        policy_logits = self.policy_head(features)
        # tanh bounds the estimate between -1 and +1, matching the loss/draw/win
        # targets. This is an expected outcome estimate, not a win probability.
        value = torch.tanh(self.value_head(features))
        return policy_logits, value

    def predict(self, board_np):
        """Predict one canonical board, returning a NumPy policy and Python value.

        The policy contains probabilities for all nine cells, including occupied
        cells. MCTS removes illegal moves and renormalizes the remaining entries.
        """
        # eval selects inference behavior for layers such as dropout or batch
        # normalization. Neither is used here, but the mode is explicit. eval
        # does not disable gradients; no_grad below handles that separately.
        self.eval()
        with torch.no_grad():
            # Search calls this many times without learning. Skipping the
            # gradient graph saves work and allows conversion back to NumPy.
            # unsqueeze adds a batch axis: one board changes from (9,) to (1, 9).
            board = torch.tensor(board_np, dtype=torch.float32).unsqueeze(0)
            logits, value = self(board)  # nn.Module routes this call through forward().
            # dim=1 normalizes across moves. [0] removes the single-board batch
            # axis; cpu makes the tensor suitable for conversion to a NumPy array.
            policy = F.softmax(logits, dim=1)[0].cpu().numpy()
        # item extracts the only value from shape (1, 1); MCTS uses plain numbers.
        return np.asarray(policy), float(value.item())
