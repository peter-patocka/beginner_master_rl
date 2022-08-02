import numpy as np
from IPython import display
from matplotlib import pyplot as plt

from envs import Maze

env = Maze()
state = env.reset()
DISPLAY_GRAPHIC = True


def random_policy(state):
    return np.array([0.25] * 4)


action_probabilities = random_policy(state)

env.reset()
done = False
img = plt.imshow(env.render(mode='rgb_array'))
while not done:
    action = np.random.choice(range(4), 1, p=action_probabilities)
    next_state, reward, done, info = env.step(action)
    print(f"We achieved a reward of {reward} by taking action {action} in state {next_state}. Info: {info}")
    img.set_data(env.render(mode='rgb_array'))
    plt.axis('off')
    if DISPLAY_GRAPHIC:
        display.display(plt.gcf())
        display.clear_output(wait=True)
env.close()
