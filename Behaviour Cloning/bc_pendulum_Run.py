import warnings
import numpy as np 
import gym
import torch
import torch.nn as nn
import torch.functional as F
from matplotlib import pyplot as plt
import pickle 
import imageio 

class MLP(nn.Module):
    def __init__(self, input_dim, size=32):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,size),
            nn.ReLU(), 
            nn.Linear(size,size),
            nn.ReLU() 
        )
    def forward(self,x):
        x = self.net(x)
        return x
    
class RegNet(MLP):
    def __init__(self, input_dim , size, action_dim):
        super(RegNet, self).__init__(input_dim, size)
        self.decoder = nn.Linear(size, action_dim)
    def forward(self,x):
        x = self.net(x)
        x = self.decoder(x)
        return x

state_dim = 3
action_dim = 1 
model = RegNet(state_dim, 32, action_dim)
model.load_state_dict(torch.load("regnet_weights.pth"))
model.eval()

print("Model weights loaded successfully!")

test_input = torch.tensor([[-0.9602126, 0.27927014, 0.732989  ]]) 
print(test_input)
with torch.no_grad():
    predicted_action = model(test_input)

print("Test Input:", test_input.numpy())
print("Predicted Action:", predicted_action.numpy())

print("Model predicting ok!")

MAX_STEP = 1000

env_name='Pendulum-v1'
env=gym.make(env_name, render_mode='human')
a = env.reset()
obs = a[0]
# print(a[0])
# exit()

REWARD = 0
for step in range(MAX_STEP):
    
    state = torch.tensor(obs, dtype=torch.float)
    with torch.no_grad():
        action = model(state).detach().numpy()

    # print(action)
    # exit()
    obs, reward, done, trunc,_ = env.step(action)
    # print(obs)
    REWARD += reward
    
    print(f"STEP {step}: Reward: {reward} Total Reward: {REWARD}")
    env.render()

print(f"Total Reward: {REWARD}")