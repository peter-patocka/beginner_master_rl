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


def policy(state, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))


def n_step_sarsa(action_values, policy, episodes, alpha=0.2, gamma=0.99, epsilon=0.2, n=8):

    for episode in range(1, episodes + 1):
        state = env.reset()
        action = policy(state, epsilon)
        transitions = []
        done = False
        t = 0

        while t-n < len(transitions):

            # Execute an action in the environment
            if not done:
                next_state, reward, done, _ = env.step(action)
                next_action = policy(next_state, epsilon)
                transitions.append([state, action, reward])

            # Update q-value estimates
            if t >= n:
                # G = rl + gamma * r2 + gamma^2 * r3 + ... + gamma^n * Q(Sn, An)
                G = (1 - done) * action_values[next_state][next_action]  # True=0, so final value is 0
                for state_t, action_t, reward_t in reversed(transitions[t-n:]):
                    G = reward_t + gamma * G

                action_values[state_t][action_t] += alpha * (G - action_values[state_t][action_t])

            t += 1
            state = next_state
            action = next_action


n_step_sarsa(action_values, policy, 1000)

print(f"Execution time: {time.time() - start_time} seconds")

plot_action_values(action_values)

# plot_policy(action_values, frame)

test_agent(env, policy)
