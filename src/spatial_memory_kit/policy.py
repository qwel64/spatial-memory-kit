import numpy as np


class ExplorationPolicy:
    """Stochastic motor bouts + a local contact reflex, NOT a learned policy."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.action = 3
        self.remaining = 0

    def act(self, observation):
        if observation['body'][:2].any():
            self.action = int(self.rng.choice([1, 2]))
            self.remaining = int(self.rng.integers(3, 9))
        elif self.remaining <= 0:
            self.action = int(self.rng.choice(4, p=[.64, .16, .16, .04]))
            self.remaining = int(self.rng.integers(1, 11 if self.action == 0 else 6))
        self.remaining -= 1
        return self.action
