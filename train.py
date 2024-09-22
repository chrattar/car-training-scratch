from agent import DQNAgent
from environment import Car
from render import draw_track, render_car
import pygame
import numpy as np

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

def train():
    for episode in range(1000):
        car.reset()  # Reset the car's position and state for each episode
        state = np.array([car.x, car.y, car.angle, car.speed])
        done = False
        total_reward = 0
        step_count = 0
        
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
                reward -= 100  # Large penalty for going out of bounds
                done = True  # End the episode

            # Store the experience and train the agent
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            step_count += 1  # Increment step count

            clock.tick(60)  # Control the speed of the simulation

        # Print total reward for the episode
        print(f"Episode {episode}, Total Reward: {total_reward}")

        # Decay epsilon to reduce exploration over time
        agent.decay_epsilon()

        # Save the model every 100 episodes
        if episode % 100 == 0:
            agent.save_model(f"car_dqn_{episode}.pth")

# Start the training loop
train()

# Save the final model after training
agent.save_model("car_dqn_final.pth")
pygame.quit()
print("Training finished.")
