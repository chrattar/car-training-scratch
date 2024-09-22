import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the Q-network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()

            q_values = self.model(torch.FloatTensor(state))
            target_f = q_values.clone()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Save the model's weights
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    # Load the model's weights
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set the model to evaluation mode
