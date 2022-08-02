import numpy as np


def random_policy():
    return np.array([0.25] * 4)


action_probabilities = random_policy()
print(f"Random policy={action_probabilities}")
policy = np.random.choice(range(4), 1, p=action_probabilities)
print(f"Random policy={random_policy()}")
