from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy.random as rd
from time import sleep
import gym
env = gym.make('LunarLander-v2')
from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()
observation = env.reset()
cum_reward = 0
frames = []
for t in range(1000):
    frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
for frame in frames:
    plt.imshow(frame)
    plt.show()
    sleep(0.01)
    clear_output(wait=True)
