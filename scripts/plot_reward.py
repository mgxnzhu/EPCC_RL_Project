import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def act_lim(action, bound):
    return (action > bound[0]) & (action < bound[1])

def reward_hist(reward, action, bound):
    reward_part = reward[act_lim(action, bound)]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.hist(reward_part, bins=50)
    ax.set_title(filename[:-4]+"+[%2.1f,%2.1f]"%bound)
    ax.set_xlabel("reward")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(filename[:-4]+"+[%2.1f,%2.1f].pdf"%bound)

for filename in os.listdir('./'):
    if os.path.splitext(filename)[1] == '.npz':
        print(filename)
        data = np.load(filename)
        reward = data['r']
        action = data['a']
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(range(len(reward)), reward)
        ax.set_title(filename[:-4])
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        fig.tight_layout()
        fig.savefig(filename[:-4]+"+stepreward.pdf")
        
        for i in np.arange(0, 1, 0.1):
            bound = (i, i+0.1)
            reward_hist(reward, action, bound)