from agent import DQNAgent
from environment import Car
from render import draw_track, render_car, draw_gates
import pygame
import numpy as np
import matplotlib.pyplot as plt  # Corrected import
import csv
import os
import pandas as pd
from datetime import datetime


#Initialize Set Up
pygame.init()
screen_width, screen_height = 800, 800
outer_radius = 350
inner_radius = 250
center = (screen_width // 2, screen_height // 2)
screen = pygame.display.set_mode((screen_width, screen_height))  # Single display surface
clock = pygame.time.Clock()

# Initialize the environment and agent
car = Car()
state_size = 4  # x, y, angle, speed
action_size = 4  # Actions: accelerate, brake, left, right
agent = DQNAgent(state_size, action_size)
max_steps_per_episode = 500
episode_rewards = []

#Dataframe Logging Info
# Define the columns for the DataFrame
columns = ["Episode", "Total Reward", "Epsilon", "Gamma", "Learning Rate", "X Position", "Y Position"]

# Initialize an empty DataFrame to store the episode data
episode_log_df = pd.DataFrame(columns=columns)

def train():
    for episode in range(500):
        car.reset()  # Reset the car's position and state for each episode
        state = np.array([car.x, car.y, car.angle, car.speed])
        done = False
        total_reward = 0
        step_count = 0  # Initialize step count for this episode
        
        while not done and step_count < max_steps_per_episode:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()  # Ensure the window can close properly
                    return
            
            # Fill the screen with a background color (e.g., black)
            screen.fill((0, 0, 0))  # Black background

            # Draw the track first, then render the car
            draw_track(screen)  # Draw the track
            draw_gates(screen)
            render_car(car, screen)  # Draw the car on the screen

            # Update the display after drawing everything
            pygame.display.flip()  # Refresh the screen

            # Move the car based on the action
            action = agent.act(state)
            car.move(action)  # Apply action to move the car
            next_state = np.array([car.x, car.y, car.angle, car.speed])
            
            # Calculate the reward
            reward = car.calculate_reward()  # Calculate the reward based on the car's movement

            # Check for collisions or out-of-bounds conditions
            if car.check_collision() or car.out_bounds():
                done = True  # End the episode

            # Store the experience and train the agent
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            step_count += 1  # Increment step count

            clock.tick(60)  # Control the speed of the simulation
        
        # Log episode data into the DataFrame
        episode_log_df.loc[episode] = [episode, total_reward, agent.epsilon, agent.gamma, agent.learning_rate, car.x, car.y, max_steps_per_episode, step_count]

        # Print total reward and step count for the episode
        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Steps: {step_count}")

        # Decay epsilon to reduce exploration over time
        agent.decay_epsilon()

        # Save the model every 100 episodes
        if episode % 100 == 0:
            agent.save_model(f"car_dqn_{episode}.pth")


def plot_rewards(agent, episode_rewards):
    # Save the rewards and DQN agent data to a CSV file
    with open('.\\pygamecar\\logdata\\episode_rewards.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Gamma", "LR", "Xpos", "Ypos", "max_steps", "step_count"])
        
     # Log data for each episode from the DataFrame
        for episode in range(len(episode_rewards)):
            # Log the episode number, total reward, and agent data (epsilon, gamma, etc.)
            row = episode_log_df.loc[episode]  # Get the row from the DataFrame
            writer.writerow(row)


    # Plot the rewards after training finishes
    plt.ioff()  # Turn off interactive mode when training is finished
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards per Episode')
    plt.savefig('.\\pygamecar\\logdata\\car_reward_plt.png')  # Save the plot as a file

def save_log_to_csv():
    # Specify the directory where you want to save the CSV files
    save_dir = ".\\pygamecar\\logdata"
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the current datetime stamp
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Add a datetime column only when saving the file
    episode_log_df['Datetime'] = current_datetime  # Add a new column with the timestamp
    
    # Check if the file already exists, if so, append the data
    file_path = os.path.join(save_dir, 'episode_rewards_with_agent_data.csv')
    if os.path.exists(file_path):
        # Append the data to the existing file without writing the header again
        episode_log_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # Write the data and include the header for a new file
        episode_log_df.to_csv(file_path, mode='w', header=True, index=False)


# Start the training loop
train()

# Plot the rewards after training
plot_rewards(episode_rewards)  # Ensure the episode rewards are passed to the function

# Save the final model after training
agent.save_model("car_dqn_final.pth")
pygame.quit()
print("Training finished.")
