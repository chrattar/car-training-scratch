import pygame
import sys
from environment import Car, gates
from render import render_car, draw_track, draw_gates

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 800  # Ensure this matches the dimensions in render.py
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Racing Game")
clock = pygame.time.Clock()

# Create a Car instance
car = Car()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Handle keyboard input for car controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car.move(0)  # Accelerate
    if keys[pygame.K_DOWN]:
        car.move(1)  # Brake
    if keys[pygame.K_LEFT]:
        car.move(2)  # Turn left
    if keys[pygame.K_RIGHT]:
        car.move(3)  # Turn right

    # Update game state
    #reward = car.calculate_reward()
    #print(f"Reward: {reward}")

    # Check for collisions or out-of-bounds and reset if necessary
    if car.check_collision() or car.out_bounds():
        print("Collision detected. Resetting...")
        car.reset()

    # Draw everything
    screen.fill((0, 0, 0))  # Fill the screen with black for contrast
    draw_track(screen)
    draw_gates(screen)
    render_car(car, screen)

    # Update the display
    pygame.display.flip()
    clock.tick(60)  # Run at 60 FPS
