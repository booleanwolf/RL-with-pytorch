import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import gym
import torch
import torch.nn as nn
import torch.functional as F
from matplotlib import pyplot as plt
import pickle 
import imageio 

env_name='Pendulum-v1'
env=gym.make(env_name, render_mode='human')
env.reset()

step = 0

print(env.action_space) 
print(env.observation_space)
exit()
while True:
    step += 1
    if step >= 100:
        break 

    action = env.action_space.sample() 
    obs, reward, done, trunc,_ = env.step(action)
    print(obs) 

    
    env.render()
