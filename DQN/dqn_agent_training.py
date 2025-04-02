import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network and Target Network
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model()

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([i[0] for i in minibatch]).squeeze(1).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).squeeze(1).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)
        
        # Current Q values
        curr_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        next_Q = self.target_model(next_states).detach().max(1)[0]
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q
        
        # Loss
        loss = F.mse_loss(curr_Q, expected_Q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def main():
    # Create the CartPole environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    
    # Number of episodes to run
    n_episodes = 1000
    
    # For plotting metrics
    all_rewards = []
    
    for e in range(n_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(500):  # Max time steps in an episode
            # Uncomment to render the environment (slows training)
            # env.render()
            
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # End episode if done
            if done:
                print(f"Episode: {e+1}/{n_episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
                
            # Train the agent with batch of experiences
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Update target model every 10 episodes
        if e % 10 == 0:
            agent.update_target_model()
            
        all_rewards.append(total_reward)
        
        # Consider the environment solved if average reward over 100 episodes is 195+
        if len(all_rewards) > 100 and np.mean(all_rewards[-100:]) > 195:
            print(f"Environment solved in {e+1} episodes!")
            agent.save("dqn_cartpole.pt")
            break
    
    env.close()

if __name__ == "__main__":
    main()