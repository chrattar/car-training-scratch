import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 64)  # Input to hidden layer
        self.fc2 = nn.Linear(64, 64)  # Hidden to hidden layer
        self.fc3 = nn.Linear(64, action_space_size)  # Hidden to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, action_space_size, state_space_size, learning_rate, discount_factor, epsilon_start, epsilon_min, epsilon_decay):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize the Q-network
        self.q_network = QNetwork(state_space_size, action_space_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Loss function

    def select_action(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Random action (explore)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()  # Best action (exploit)

    def learn(self, state, action, reward, next_state, done):
        # Convert states to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_space_size)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Compute Q-values for the current state
        q_values = self.q_network(state_tensor)
        q_value = q_values[0, action]

        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (self.discount_factor * max_next_q_value * (1 - done))

        # Ensure both q_value and target_q_value are the same size
        target_q_value = torch.tensor([target_q_value])  # Convert to tensor with shape (1,)

        # Compute loss
        loss = self.criterion(q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.q_network.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the model from a file."""
        self.q_network.load_state_dict(torch.load(file_path))
        self.q_network.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {file_path}")
