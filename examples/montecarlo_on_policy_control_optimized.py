import numpy as np
import matplotlib.pyplot as plt
import time

from envs import Maze
from utils import plot_policy, plot_action_values, test_agent

start_time = time.time()

env = Maze()

frame = env.render(mode='rgb_array')
plt.axis('off')
plt.imshow(frame)

print(f"Observation space shape: {env.observation_space.nvec}")
print(f"Number of actions: {env.action_space.n}")

action_values = np.zeros((5, 5, 4))

# plot_action_values(action_values)


def policy(state, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))


action = policy((0, 0), epsilon=0.5)
# print(f"Action taken in state (0, 0): {action}")
# plot_policy(action_values, frame)


def on_policy_mc_control(policy, action_values, episodes, gamma=0.99, epsilon=0.2, alpha=0.2):

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        transitions = []

        while not done:
            action = policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            transitions.append([state, action, reward])
            state = next_state

        G = 0

        for state_t, action_t, reward_t in reversed(transitions):
            G = reward_t + gamma * G

            qsa = action_values[state_t][action_t]
            action_values[state_t][action_t] += alpha * (G - qsa)


on_policy_mc_control(policy, action_values, episodes=100)

print(f"Execution time: {time.time() - start_time} seconds")

plot_action_values(action_values)

# plot_policy(action_values, frame)

test_agent(env, policy)
