# loggingtasks.py

import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

def initialize_log_df():
    columns = ["Episode", "Total Reward", "Epsilon", "Gamma", "LR", "Xpos", "Ypos", "max_steps", "step_count"]
    return pd.DataFrame(columns=columns)

def log_episode(df, episode, total_reward, agent, car, max_steps, step_count):
    df.loc[episode] = [episode, total_reward, agent.epsilon, agent.gamma, agent.learning_rate, car.x, car.y, max_steps, step_count]

def plot_rewards(episode_rewards, episode_log_df):
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
    plt.savefig('.\\pygamecar\\logdata\\car_reward_plt.png')

def save_log_to_csv(episode_log_df):
    save_dir = ".\\pygamecar\\logdata"
    os.makedirs(save_dir, exist_ok=True)
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    episode_log_df['Datetime'] = current_datetime
    
    file_path = os.path.join(save_dir, 'ep_rewards_timestamp.csv')
    if os.path.exists(file_path):
        episode_log_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        episode_log_df.to_csv(file_path, mode='w', header=True, index=False)
