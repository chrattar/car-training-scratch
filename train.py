# train.py

from agent import DQNAgent
from environment import Car
from render import draw_track, render_car, draw_gates
import pygame
import numpy as np
from loggingtasks import initialize_log_df, log_episode, plot_rewards, save_log_to_csv
from plotter import EpisodePlotter
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
num_rays = 10  # Number of rays for ray casting
ray_length = 300  # Length of rays
state_size = 4 + num_rays  # x, y, angle, speed, plus ray distances
action_size = 4  # Actions: accelerate, brake, left, right
agent = DQNAgent(state_size, action_size)
max_steps_per_episode = 4000
update_frequency = 2
episode_rewards = []
plotter = EpisodePlotter()
# Initialize logging DataFrame
episode_log_df = initialize_log_df()

def train():
    num_episodes = 2000
    agent.clear_memory()

    for episode in range(num_episodes):
        car.reset()
        ray_results = car.cast_rays(num_rays, ray_length)
        state = np.array([car.x, car.y, car.angle, car.speed] + ray_results)
        
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    plotter.close()
                    return
            
            screen.fill((0, 0, 0))
            draw_track(screen)
            draw_gates(screen)
            render_car(car, screen)
            pygame.display.flip()

            action = agent.act(state)
            car.move(action)
            ray_results = car.cast_rays(num_rays, ray_length)
            next_state = np.array([car.x, car.y, car.angle, car.speed] + ray_results)
            reward = car.calculate_reward()
            done = car.check_collision() or car.out_bounds()
            agent.remember(state, action, reward, next_state, done)
            
            if step % update_frequency == 0:
                agent.train()

            state = next_state
            total_reward += reward

            if done:
                break
            clock.tick(120)      
        agent.decay_epsilon()
        
        # Log episode data
        log_episode(episode_log_df, episode, total_reward, agent, car, max_steps_per_episode, step+1)
        episode_rewards.append(total_reward)
        plotter.update(episode, total_reward)
        
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {step+1}, Epsilon: {agent.epsilon:.2f}")

        # Save the model every 1000 episodes
        if episode % 100 == 0:
            agent.save_model(f"car_dqn_{episode}.pth")
    plotter.close()
# MAIN()
train()
plot_rewards(episode_rewards, episode_log_df)
agent.save_model("car_dqn_final.pth")
save_log_to_csv(episode_log_df)
pygame.quit()
print("Training finished.")
