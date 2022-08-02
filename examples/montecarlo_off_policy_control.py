import numpy as np
import matplotlib.pyplot as plt

from envs import Maze
from utils import plot_policy, plot_action_values, test_agent

env = Maze()

frame = env.render(mode='rgb_array')
plt.axis('off')
plt.imshow(frame)

print(f"Observation space shape: {env.observation_space.nvec}")
print(f"Number of actions: {env.action_space.n}")

action_values = np.full((5, 5, 4), -100)
action_values[4, 4, :] = 0.

# plot_action_values(action_values)


def target_policy(state):
    av = action_values[state]
    return np.random.choice(np.flatnonzero(av == av.max()))


# action = target_policy((0, 0))
# print(f"Action taken (0, 0): {action}")
# plot_policy(action_values, frame)


def exploratory_policy(state, epsilon=0.):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))


# action = exploratory_policy((0, 0), epsilon=0.5)
# print(f"Action taken (0, 0): {action}")


def off_policy_mc_control(action_values, target_policy, exploratory_policy, episodes, gamma=0.99, epsilon=0.2):
    csa = np.zeros((5, 5, 4))

    for episode in range(1, episodes + 1):
        G = 0
        W = 1
        state = env.reset()
        done = False
        transitions = []

        while not done:
            action = exploratory_policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            transitions.append([state, action, reward])
            state = next_state

        for state_t, action_t, reward_t in reversed(transitions):
            G = reward_t + gamma * G
            csa[state_t][action_t] += W
            qsa = action_values[state_t][action_t]
            action_values[state_t][action_t] += (W / csa[state_t][action_t]) * (G - qsa)

            if action_t != target_policy(state_t):
                break;

            W = W * 1. / (1 - epsilon + epsilon / 4)


off_policy_mc_control(action_values, target_policy, exploratory_policy, 1000, epsilon=0.3)

# plot_action_values(action_values)
# plot_policy(action_values, frame)
test_agent(env, target_policy)
