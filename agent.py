# agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Neural Network for Deep Q-Learning
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Input layer: state_size -> 128 neurons
        self.fc2 = nn.Linear(128, 64)          # Hidden layer: 128 neurons -> 64 neurons
        self.fc3 = nn.Linear(64, action_size)  # Output layer: 64 neurons -> action_size neurons

    def forward(self, state):
        x = F.relu(self.fc1(state))  # Pass through input layer and apply ReLU
        x = F.relu(self.fc2(x))      # Pass through hidden layer and apply ReLU
        q_values = self.fc3(x)       # Output layer: Q-values for each action
        return q_values

# DQN Agent class for training and interaction
class Agent:
    def __init__(self, state_size, action_size, device):  # Accept device as an argument
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = device  # Set device from argument
        self.model = DQNAgent(state_size, action_size).to(self.device)  # Move model to GPU
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Add batch dimension and move to GPU
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                 target += self.gamma * torch.max(self.model(next_state.unsqueeze(0))).item()


            # Debugging prints
            #print(f"Replay - State shape: {state.shape}")
            #print(f"Replay - Next state shape: {next_state.shape}")

            # Get predicted Q-values for current state
            target_f = self.model(state.unsqueeze(0))  # Add batch dimension
            #print(f"Shape of state before unsqueeze: {state.shape}")

            # Detach the target to prevent gradients from backpropagating through it
            target_f = target_f.clone().detach()
            #print(f"Shape of q_values: {q_values.shape}")

            # Update only the Q-value for the selected action
            target_f[0][action] = target  # Now indexing correctly

            states.append(state)
            targets.append(target_f)

        states = torch.stack(states)
        targets = torch.cat(targets)  # Concatenate targets

        # Compute loss in batch
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
