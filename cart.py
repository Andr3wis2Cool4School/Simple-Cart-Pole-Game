# The first thing is to import things


import gym 
import matplotlib.pyplot as plt
import random 
import warnings
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')

"""
By Yifeng Luo


This is a little demo of reinforcement learning

Q learning 

Please enjoy :)


Notes: 
The Hyper parameters

1. alpha is the learning rate
2. gamma is the discounted reward
3. epsilon is the expore probabilites


Code Structure:

1. Class Setting: store the hyper parameters
2. Main function  

In the main function, alpha == 0.3, gamma == 0.9, epsilon == 0.9
"""


class sett:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {}

        self.actions = (0, 1) # which is an action tuple; where 0 is left, 1 is right
        self.env = gym.make("CartPole-v1")
    
    def qvalues(self, state):
        return [self.Q.get((state, a), 0) for a in self.actions]

    def probs(self, v, eps=1e-4):
        v = v - v.min() + eps
        v = v/v.sum()
        return v

    def discretize(self, x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))


def main():
    setting = sett(0.3, 0.9, 0.9)
    
    for epoch in range(10000):
        obs = setting.env.reset()
        done = False
        while not done:
            s = setting.discretize(obs)

            if random.random() < setting.epsilon:
                v = setting.probs(np.array(setting.qvalues(s)))
                a = random.choices(setting.actions, weights=v)[0]
            else:
                a = np.random.randint(setting.env.action_space.n)

            obs, rew, done, info =setting.env.step(a)
            ns = setting.discretize(obs)
            setting.Q[(s, a)] = setting.Q.get((s, a), 0) + setting.alpha * (rew + setting.gamma * max(setting.qvalues(ns)) - setting.Q.get((s, a), 0))

    obs = setting.env.reset()
    done = False
    image_append = []

    while not done: 
        s = setting.discretize(obs)
        current_img = setting.env.render(mode="rgb_array")
        current_img = Image.fromarray(current_img)
        image_append.append(current_img)

        v = setting.probs(np.array(setting.qvalues(s)))
        a = random.choices(setting.actions, weights=v)[0]
        obs, _, done, _ = setting.env.step(a)

    image_append[0].save('./out2.gif', save_all=True, append_images=image_append)
    setting.env.close()


if __name__ == '__main__':
    main()
