from agent import DQNAgent
from environment import Car
from render import draw_track, render_car, draw_gates
import pygame
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from datetime import datetime

# Initialize Set Up
pygame.init()
screen_width, screen_height = 800, 800
outer_radius = 350
inner_radius = 250
center = (screen_width // 2, screen_height // 2)
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Init car and env
car = Car()
state_size = 4  # x, y, angle, speed
action_size = 4  # Actions: accelerate, brake, left, right
agent = DQNAgent(state_size, action_size)
max_steps_per_episode = 4000  # Reduced from 2000 to 1000
update_frequency = 2
episode_rewards = []

# df Log info
columns = ["Episode", "Total Reward", "Epsilon", "Gamma", "LR", "Xpos", "Ypos", "max_steps", "step_count"]
episode_log_df = pd.DataFrame(columns=columns)

def train():
    num_episodes = 2000  # Increased from 500 to 1000

    for episode in range(num_episodes):
        car.reset()
        state = np.array([car.x, car.y, car.angle, car.speed])
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            screen.fill((0, 0, 0))
            draw_track(screen)
            draw_gates(screen)
            render_car(car, screen)
            pygame.display.flip()

            action = agent.act(state)
            car.move(action)
            next_state = np.array([car.x, car.y, car.angle, car.speed])
            reward = car.calculate_reward()
            done = car.check_collision() or car.out_bounds()

            agent.remember(state, action, reward, next_state, done)
            
            if step % update_frequency == 0:
                agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

            clock.tick(60)
        agent.decay_epsilon()
        
        # Log episode data
        episode_log_df.loc[episode] = [episode, total_reward, agent.epsilon, agent.gamma, agent.learning_rate, car.x, car.y, max_steps_per_episode, step+1]
        episode_rewards.append(total_reward)
        
        print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Steps: {step+1}, Epsilon: {agent.epsilon:.2f}")

        # Save the model 100 ep
        if episode % 100 == 0:
            agent.save_model(f"car_dqn_{episode}.pth")

def plot_rewards(agent, episode_rewards):
    with open('.\\pygamecar\\logdata\\episode_rewards_dfd.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Gamma", "LR", "Xpos", "Ypos", "max_steps", "step_count"])
        for episode in range(len(episode_rewards)):
            row = episode_log_df.loc[episode]
            writer.writerow(row)

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Total Rewards per Episode')
    plt.style.use('ggplot')
    plt.grid(True)
    
    # Roll average
    window_size = 50
    rolling_mean = pd.Series(episode_rewards).rolling(window=window_size).mean()
    plt.plot(rolling_mean, color='red', label=f'{window_size}-episode Rolling Average')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('.\\pygamecar\\logdata\\car_reward_plt_dfd.png')

def save_log_to_csv():
    save_dir = ".\\pygamecar\\logdata"
    os.makedirs(save_dir, exist_ok=True)
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    episode_log_df['Datetime'] = current_datetime
    
    file_path = os.path.join(save_dir, 'episode_rewards_with_agent_data.csv')
    if os.path.exists(file_path):
        episode_log_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        episode_log_df.to_csv(file_path, mode='w', header=True, index=False)

# MAIN()
train()
plot_rewards(agent, episode_rewards)
agent.save_model("car_dqn_final.pth")
save_log_to_csv()
pygame.quit()
print("Training finished.")
