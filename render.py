#render.py

import pygame
import numpy as np
from config import screen, SCREEN_HEIGHT, SCREEN_WIDTH
from Box2D import b2PolygonShape

def initialize_screen():
    pygame.init()
    return pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

def draw_car(car, screen):
    position = car.position
    angle = car.angle
    x, y = position[0] * 10, SCREEN_HEIGHT - position[1] * 10  # Convert Box2D to Pygame (1 meter = 10 pixels)
      # Debugging output to check car's position
    print(f"Car position: ({position[0]:.2f}, {position[1]:.2f}) meters -> ({x:.2f}, {y:.2f}) pixels")
    print(f"Car angle: {angle:.2f} radians")

    car_color = (255, 0, 0)  # Red color for the car
    car_width, car_height = 15, 30  # Size in pixels

    # Create a surface for the car with a transparent background
    car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    car_surface.fill(car_color)

    # Rotate the car surface according to the car's angle
    rotated_car_surface = pygame.transform.rotate(car_surface, angle * (180.0 / np.pi))

    # Get the rectangle around the rotated car
    car_rect = rotated_car_surface.get_rect(center=(x, y))
    
    # Blit (draw) the rotated car onto the screen at the correct position
    screen.blit(rotated_car_surface, car_rect.topleft)

def draw_environment(environment, screen):
    # Green background for the environment
    screen.fill((0, 255, 122))  

    # Draw the black circular track (outer and inner boundaries)
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    outer_radius = 400
    inner_radius = 170

    # Draw the outer boundary of the track as a black circle
    pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), outer_radius)

    # Draw the inner boundary of the track as a green circle to create a ring effect
    pygame.draw.circle(screen, (0, 255, 122), (center_x, center_y), inner_radius)

    # Draw white lines at intervals of 100 pixels
    for y in range(100, 1080, 102):
        pygame.draw.line(screen, (255, 255, 255), (0, y), (SCREEN_WIDTH, y), 2)

def update_screen():
    pygame.display.flip()

def clear_screen(screen):
    screen.fill((0, 255, 122))  # Fill screen with green color to clear
