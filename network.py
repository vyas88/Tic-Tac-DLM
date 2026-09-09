import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(9, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, 9)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.body(x)
        policy_logits = self.policy_head(features)
        # +1 means the current player is winning, -1 means losing.
        value = torch.tanh(self.value_head(features))
        return policy_logits, value

    def predict(self, board_np):
        self.eval()
        with torch.no_grad():
            board = torch.tensor(board_np, dtype=torch.float32).unsqueeze(0)
            logits, value = self(board)
            policy = F.softmax(logits, dim=1)[0].cpu().numpy()
        return np.asarray(policy), float(value.item())
