import numpy as np

class MCTSNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.action = None
        self.ucb = np.inf  # Initially set to infinity to ensure this node is selected first


    def expand(self, priors):
        for i, prior in enumerate(priors):
            child = MCTSNode(self, prior)
            child.action = i
            self.children.append(child)

class MCTS:
    def __init__(self, size_x, size_y, cur_pos, distribution, obstacles, num_iterations, c_param, max_rollout_steps=10):
        self.size_x = size_x
        self.size_y = size_y
        self.cur_pos = cur_pos
        self.distribution = distribution
        self.num_iterations = num_iterations
        self.c_param = c_param
        self.max_rollout_steps = max_rollout_steps
        self.actions = {
            0: np.array([0, -1]),  # up
            1: np.array([-1, 0]),  # left
            2: np.array([0, 1]),  # down
            3: np.array([1, 0]),  # right
        }
        self.total_visits = 0
        self.root = MCTSNode(None, 1.0)  # Create the root node with None as the parent

    def compute_ucb(self, node):
        # Only compute UCB if the node is being visited for the first time or its visit count changes
        if node.visits == 0 or node.visits != node.parent.visits:
            node.ucb = node.value / (1 + node.visits) + self.c_param * node.prior * np.sqrt(self.total_visits) / (1 + node.visits)
        return node.ucb

    def compute_priors(self, cur_pos):
        action_rewards = []
        for action, delta in self.actions.items():
            next_pos = cur_pos + delta
            if 0 <= next_pos[0] < self.size_x and 0 <= next_pos[1] < self.size_y:
                action_rewards.append(self.distribution[next_pos[0], next_pos[1]])
            else:
                action_rewards.append(-np.inf)  # Discourage moving off the grid
        # Transform action rewards into probabilities
        max_reward = max(action_rewards)
        action_rewards = [np.exp(r - max_reward) if r != -np.inf else 0 for r in action_rewards]  # Stable softmax
        total = sum(action_rewards)
        action_probabilities = [r / total if total != 0 else 1.0 / len(self.actions) for r in action_rewards]
        return action_probabilities

    def select_child(self, node):
        best_score = -np.inf
        best_child = None
        for child in node.children:
            score = self.compute_ucb(child)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def tree_search(self, node):
        path = [node]
        cur_pos = self.cur_pos.copy()
        while len(path[-1].children) != 0:
            node = self.select_child(path[-1])
            cur_pos += self.actions[node.action]
            if not (0 <= cur_pos[0] < self.size_x and 0 <= cur_pos[1] < self.size_y):
                break
            path.append(node)
        if path[-1].visits == 0:
            reward = self.rollout(path[-1], cur_pos)
            return path[-1], reward
        else:
            priors = self.compute_priors(cur_pos)
            path[-1].expand(priors)
            node = self.select_child(path[-1])
            reward = self.rollout(node, cur_pos)
            return node, reward

    def rollout(self, node, cur_pos):
        num_steps = self.max_rollout_steps
        reward = 0
        cur_pos = cur_pos.copy()
        for _ in range(num_steps):
            action = np.random.choice(len(self.actions))  # Random action selection
            cur_pos += self.actions[action]
            if not (0 <= cur_pos[0] < self.size_x and 0 <= cur_pos[1] < self.size_y):
                break
            reward += self.distribution[cur_pos[0], cur_pos[1]]
        return reward


    def get_best_action(self, cur_pos):
        action_probabilities = self.compute_priors(cur_pos)
        return np.random.choice(len(self.actions), p=action_probabilities)

    def backpropagate(self, path, reward):
        for node in reversed(path):
            node.visits += 1
            node.value += reward
            reward *= 0.99  # Discount factor for multi-step ahead reward

    def simulate(self):
        for _ in range(self.num_iterations):
            node, reward = self.tree_search(self.root)
            path = [node]
            while node is not self.root:
                node = node.parent
                path.append(node)
            self.backpropagate(path, reward)
            self.total_visits += 1
        return self.select_child(self.root).action

