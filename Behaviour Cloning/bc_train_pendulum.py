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

data_path = "expert_data/Pendulum-v1_10_-130.pkl"

with open(data_path, "rb") as f:
    data_good = pickle.load(f)
print('expert data loaded')
# print(data_good)
data_good=data_good[:20]

good_obs=[]
good_acts=[] 
for traj in data_good: 
    s,a,r=traj   
    good_obs.append(s)
    good_acts.append(a) 

states=np.vstack(good_obs)
actions=np.vstack(good_acts)
print(states)
print(actions)
print('X:',states.shape,' y:', actions.shape)
# exit()

data_loader = torch.utils.data.DataLoader( list(zip(states, actions)), batch_size=64, shuffle=True)

batch=next(iter(data_loader))
states,actions = batch
# print(states, actions)
print(states.shape, actions.shape)
print("Loaded into dataloader")

action_dim=actions.shape[1]
state_dim=states.shape[1]
print(state_dim, action_dim)


model = RegNet(state_dim, 32, action_dim)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

loss_list = []
test_loss = []
batch_size = 256
n_epoch = 1_000
 
for itr in range(0, n_epoch+1):
    total_loss = 0
    b=0
    for batch_states, batch_actions in data_loader: 
        y_pred = model(batch_states)
        loss   = criterion(y_pred, batch_actions) 
        total_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b += 1 
        
        loss_list.append(loss.item())
        
    if itr % (n_epoch//20)==0:
        print(f'Epoch {itr} Loss: {total_loss/b:.3f}')


torch.save(model.state_dict(), "regnet_weights.pth")
print("Model weights saved as regnet_weights.pth")


