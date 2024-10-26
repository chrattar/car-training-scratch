import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os

# Load dir and name
current_working_dir = os.getcwd()
print("Current working directory:", current_working_dir)
file_name = 'episode_rewards_dfd.csv'
file_path = "C:\\stentor\\pyLLM\\pygamecar\\logdata\\episode_rewards_dfd.csv"
print("File exists:", os.path.exists(file_path))
print("Constructed file path:", file_path)
try:
    df = pd.read_csv(file_path)
    print("CSV file read successfully")
    print(df.head())
    print("Columns:", df.columns)
    
    avg_reward = df['Total Reward'].mean()
    stepcount = df['step_count'].mean()
    print(f"Avg Rewards: {avg_reward:.2f}, Avg Steps: {stepcount:.2f}")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Episode'], df['Total Reward'])
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.xlabel('Episode')
    plt.xticks(np.arange(0, df['Episode'].max() + 1, step=max(1, df['Episode'].max() // 10)))
    plt.ylabel('Total Reward')
    plt.show()

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {file_path} is empty.")
except pd.errors.ParserError:
    print(f"Error: Unable to parse {file_path}. Make sure it's a valid CSV file.")
except KeyError as e:
    print(f"Error: The column {e} was not found in the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
