import numpy as np


class Node:
    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net, c_puct=1.5, n_simulations=40):
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def run(self, game, add_noise=True):
        root = Node(1.0)
        for _ in range(self.n_simulations):
            state = game.clone()
            node = root
            path = [node]

            while node.children:
                action, node = self._select_child(node)
                state.step(action)
                path.append(node)

            value = self._expand_and_evaluate(node, state)
            if node is root and add_noise and node.children:
                # Root noise makes self-play explore different opening moves.
                noise = np.random.dirichlet([0.3] * len(node.children))
                for child, noise_value in zip(node.children.values(), noise):
                    child.prior = 0.75 * child.prior + 0.25 * noise_value

            # Values change viewpoint at every level because turns alternate.
            for visited_node in reversed(path):
                visited_node.visit_count += 1
                visited_node.value_sum += value
                value = -value

        visits = np.zeros(9, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() > 0:
            visits /= visits.sum()
        return visits

    def _select_child(self, parent):
        best_action = None
        best_child = None
        best_score = -float("inf")
        for action, child in parent.children.items():
            # Q is exploitation; the second term is prior-weighted exploration.
            # A child's Q belongs to the next mover, so negate it for the parent.
            q_value = -child.q_value()
            explore = self.c_puct * child.prior * np.sqrt(parent.visit_count)
            score = q_value + explore / (1 + child.visit_count)
            if score > best_score:
                best_action, best_child, best_score = action, child, score
        return best_action, best_child

    def _expand_and_evaluate(self, node, state):
        winner = state.check_winner()
        if winner is not None:
            return winner * state.current_player

        # AlphaZero uses the network value here, with no random rollouts.
        policy, value = self.net.predict(state.get_canonical())
        legal_moves = state.legal_moves()
        legal_policy = policy[legal_moves]
        legal_policy /= legal_policy.sum()
        for action, prior in zip(legal_moves, legal_policy):
            node.children[action] = Node(prior)
        return value
