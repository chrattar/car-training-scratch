import pygame
from environment import Environment
from agent import Agent
from render import draw_car, draw_environment, update_screen, clear_screen
from config import screen, NUM_EPISODES, should_render, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, EPSILON_MIN, EPSILON_DECAY

def train(agent, environment):
    for episode in range(NUM_EPISODES):
        state = environment.reset()  # Reset environment
        done = False
        frame_count = 0

        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.select_action(state)
            next_state, reward, done = environment.step(action)  # Use environment's step method

            agent.learn(state, action, reward, next_state, done)

            # Increment frame counter and render if needed
            frame_count += 1
            if should_render(frame_count):
                clear_screen(screen)
                draw_environment(environment, screen)
                draw_car(environment.car, screen)
                update_screen()

            state = next_state

            #print(f"Episode: {episode}, Frame: {frame_count}, State: {state}")
            print(f"Episode: {episode} Action: {action}, Reward: {reward} State: {state}")

            # Ensure the loop continues
            pygame.time.wait(10)  # Add a slight delay to avoid maxing out the CPU

if __name__ == "__main__":
    # Initialize Pygame and the display
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("Car Simulation")

    # Initialize environment and agent
    environment = Environment()
    
    action_space_size = 4  # Actions: accelerate, decelerate, turn left, turn right
    state_space_size = 5  # X, Y position, X, Y velocity, orientation

    agent = Agent(
        action_space_size=action_space_size,
        state_space_size=state_space_size,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )

    # Start training
    train(agent, environment)

    # Ensure Pygame is cleaned up properly
    pygame.quit()
