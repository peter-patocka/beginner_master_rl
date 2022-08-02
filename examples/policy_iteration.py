import numpy as np
import matplotlib.pyplot as plt
import time

from envs import Maze
from utils import plot_policy, plot_values, plot_action_values, test_agent

env = Maze()
DEBUG = False

frame = env.render(mode='rgb_array')
plt.axis('off')
plt.imshow(frame)

print(f"Observation space shape: {env.observation_space.nvec}")
print(f"Number of actions: {env.action_space.n}")

policy_probs = np.full((5, 5, 4), 0.25)


def policy(state):
    return policy_probs[state]


action_probabilities = policy((0,0))
for action, prob in zip(range(4), action_probabilities):
    print(f"Probability of taking action {action}: {prob}")


state_values = np.zeros(shape=(5,5))

# test_agent(env, policy, episodes=1)
# plot_policy(policy_probs, frame)
# plot_values(state_values, frame)


def policy_evaluation(policy_probs, state_values, theta=1e-6, gamma=0.99):
    delta = float("inf")

    while delta > theta:
        delta = 0

        for row in range(5):
            for col in range(5):
                old_value = state_values[(row, col)]
                new_value = 0.
                action_probabilities = policy_probs[(row, col)]

                for action, prob in enumerate(action_probabilities):
                    next_state, reward, _, _ = env.simulate_step((row, col), action)
                    new_value += prob * (reward + gamma * state_values[next_state])
                    if DEBUG:
                        print(f"Iterating over ({row}, {col}) and taking action {action}. new_value={new_value}")

                state_values[(row, col)] = new_value

                delta = max(delta, abs(old_value - new_value))


def policy_improvement(policy_probs, state_values, gamma=0.99):
    policy_stable = True

    for row in range(5):
        for col in range(5):
            old_action = policy_probs[(row, col)].argmax()

            new_action = None
            max_qsa = float("-inf")

            for action in range(4):
                if DEBUG:
                    print(f"Improving ({row}, {col}) by taking action {action}")
                next_state, reward, _, _ = env.simulate_step((row, col), action)
                qsa = reward + gamma * state_values[next_state]

                if qsa > max_qsa:
                    new_action = action
                    max_qsa = qsa

            action_probs = np.zeros(4)
            action_probs[new_action] = 1.
            policy_probs[(row, col)] = action_probs

            if new_action != old_action:
                policy_stable = False

    return policy_stable


def policy_iteration(policy_probs, state_values, theta=1e-6, gamma=0.99):
    policy_stable = False
    i = 0

    while not policy_stable:
        i += 1
        if DEBUG:
            print(f"Policy is still not stable, counter: {i}")
        policy_evaluation(policy_probs, state_values, theta, gamma)
        # plot_values(state_values, frame)
        policy_stable = policy_improvement(policy_probs, state_values, gamma)
        # plot_policy(policy_probs, frame)

    pass


policy_iteration(policy_probs, state_values)


# plot_values(state_values, frame)
# plot_policy(policy_probs, frame)
plot_action_values(policy_probs)

test_agent(env, policy)
