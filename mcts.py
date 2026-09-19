"""Use a policy-value network to guide Monte Carlo Tree Search (MCTS).

Each simulation selects a path, expands/evaluates a leaf, and backs its value
up the path. This implementation evaluates leaves with the network instead
of finishing them with random rollouts. Repeated visits produce a move policy
that train.py uses both to play games and to teach the policy head.
"""

import numpy as np  # Exploration noise, square roots, and visit-count arrays.


class Node:
    """Store search statistics for a position, from its player-to-move viewpoint.

    The incoming prior belongs to the move that led here. The value statistics
    belong to the player about to move here, so parent and child values have
    opposite perspectives. Boards are reconstructed on a cloned game in run().
    """

    def __init__(self, prior):
        """Start an unvisited node with its network-provided move probability."""
        self.prior = float(prior)
        self.visit_count = 0  # N: number of simulations whose path includes this node.
        self.value_sum = 0.0  # W: sum of backed-up outcomes/estimates for this player.
        self.children = {}  # action index -> child Node, populated when expanded.

    def q_value(self):
        """Return the average backed-up value Q = W/N, or zero before any visits."""
        # Zero avoids dividing by zero and gives an unexplored move a neutral
        # value estimate until a simulation supplies evidence.
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Combine learned move priors with simulated outcomes to choose moves."""

    def __init__(self, net, c_puct=1.5, n_simulations=40):
        """Set the network, exploration weight, and search budget for each move."""
        self.net = net
        self.c_puct = c_puct  # Larger values favor exploring promising unvisited moves.
        self.n_simulations = n_simulations  # Larger search budgets cost more computation.

    def run(self, game, add_noise=True):
        """Return nine normalized root visit counts without changing game.

        Call with an unfinished game and at least two simulations to obtain a
        move distribution: the first simulation only expands the root. Self-play
        uses root noise; evaluation and the desktop opponent disable it.
        """
        # Build a fresh tree for each call. The root has no incoming move, so
        # its prior of 1.0 is a placeholder, not a predicted move probability.
        root = Node(1.0)
        for _ in range(self.n_simulations):
            # Each simulation starts from the real position on its own copy.
            # Nodes retain statistics across simulations; the copied board does not.
            state = game.clone()
            node = root
            path = [node]  # Remember ancestors so the leaf value can update them.

            # Selection: follow the best exploration/exploitation score until
            # reaching an unexpanded node or an already known terminal node.
            while node.children:
                action, node = self._select_child(node)
                state.step(action)
                path.append(node)

            # Expansion adds legal children; evaluation returns a value from
            # the leaf's player-to-move viewpoint, ready for backup below.
            value = self._expand_and_evaluate(node, state)
            if node is root and add_noise and node.children:
                # The root is expanded once in a normal search, so its noise
                # is mixed in once. A Dirichlet sample is a probability vector
                # over legal moves; alpha=0.3 tends to give uneven proportions.
                # Mixing 75% learned prior with 25% noise encourages varied
                # self-play choices at every move, including varied openings.
                noise = np.random.dirichlet([0.3] * len(node.children))
                for child, noise_value in zip(node.children.values(), noise):
                    child.prior = 0.75 * child.prior + 0.25 * noise_value

            # Backup: update the leaf first, then work toward the root. A leaf
            # value of -1 means its mover loses; its parent's mover therefore
            # gets +1. Flip the sign after storing each node's own perspective.
            # This is a search-statistics update, not neural-network backpropagation.
            for visited_node in reversed(path):
                visited_node.visit_count += 1
                visited_node.value_sum += value
                value = -value

        # Only legal root actions have children, so occupied cells keep zero
        # probability. Visits summarize search effort rather than raw network
        # preference; normalizing them creates the policy target for training.
        visits = np.zeros(9, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        # The guard avoids division by zero for a terminal board or an
        # insufficient search budget; those cases leave an all-zero result.
        if visits.sum() > 0:
            visits /= visits.sum()
        return visits

    def _select_child(self, parent):
        """Return the action and child with the highest PUCT selection score."""
        best_action = None
        best_child = None
        best_score = -float("inf")  # The first legal child's score must beat this.
        for action, child in parent.children.items():
            # PUCT balances exploitation and prior-weighted exploration:
            # score = -Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child)).
            # Negate Q because the child's mover is the parent's opponent.
            q_value = -child.q_value()
            # A high prior P makes a move worth investigating. The numerator
            # grows with total search effort, while the denominator reduces
            # the bonus as this particular move receives more visits.
            explore = self.c_puct * child.prior * np.sqrt(parent.visit_count)
            score = q_value + explore / (1 + child.visit_count)
            if score > best_score:
                # Strict > keeps the first encountered move when scores tie.
                best_action, best_child, best_score = action, child, score
        return best_action, best_child

    def _expand_and_evaluate(self, node, state):
        """Return an exact terminal value, or expand legal moves using the network."""
        winner = state.check_winner()
        if winner is not None:
            # winner uses absolute X/O signs; multiplying by current_player
            # converts to this node's perspective. step() has already switched
            # turns after a winning move, so the next mover sees a loss (-1).
            # A draw stays 0. Terminal nodes need neither children nor a prediction.
            return winner * state.current_player

        # The network predicts from the same perspective used for stored values.
        # Its value replaces a random rollout, letting search evaluate a leaf
        # without simulating every remaining move to the end of the game.
        policy, value = self.net.predict(state.get_canonical())
        legal_moves = state.legal_moves()
        # The network scores all nine cells. Select only empty cells and make
        # their probabilities sum to one again. NumPy list indexing creates
        # a separate array, so this normalization does not change policy.
        legal_policy = policy[legal_moves]
        legal_policy /= legal_policy.sum()
        for action, prior in zip(legal_moves, legal_policy):
            # New children start with a prior but no outcome evidence or visits.
            node.children[action] = Node(prior)
        return value
