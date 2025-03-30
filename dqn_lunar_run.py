import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the DQN model class (must match the architecture used for training)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def run_simulation(model_path, num_episodes=5):
    # Create the environment
    env = gym.make('LunarLander-v2', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        
        while not done:
            # Render the environment
            env.render()
            
            # Get action from the model
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            
            # Take action and observe result
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    # Path to the saved model
    model_path = "dqn_land.pt"
    
    # Run simulation with the trained model
    run_simulation(model_path)

# If you haven't trained a model yet, you can run a simulation with a random agent:
def run_random_simulation(num_episodes=5):
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            env.render()
            action = env.action_space.sample()  # Random action
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
        print(f"Random Agent - Episode: {episode+1}, Total Reward: {total_reward}")
    
    env.close()

# Uncomment this line if you want to run a simulation with a random agent instead
# run_random_simulation()