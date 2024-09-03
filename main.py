import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody

# Initialize Pygame
pygame.init()

# Define the environment
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Box2D Car Racing")

# Load the car and track images
car_image = pygame.image.load('car.png')  # Replace 'car.png' with the path to your car image
car_image = pygame.transform.scale(car_image, (36, 60))  # Adjust the scale as needed
track_image = pygame.image.load('track.png')

# Create the Box2D world with zero gravity
world = b2World(gravity=(0, 0))

def create_car():
    # Define the car's physics body
    car_body_def = b2BodyDef()
    car_body_def.type = b2_dynamicBody
    car_body_def.position = (10, 10)  # Set the initial position

    car = world.CreateBody(car_body_def)
    
    # Define the car's shape in Box2D
    # Adjust shape dimensions to match car size in the simulation
    car_shape = b2PolygonShape(box=(1.8, 3.0))  # Shape in meters, not pixels
    car.CreateFixture(shape=car_shape, density=1.0, friction=0.3)

    return car

def draw_car(car):
    position = car.position
    angle = car.angle
    x, y = 946, 146  # Convert Box2D to Pygame coordinates

    # Rotate the car image according to the Box2D body's angle
    rotated_image = pygame.transform.rotate(car_image, -angle * (180.0) / np.pi)
    rect = rotated_image.get_rect(center=(x, y))
    
    # Draw the car on the screen
    screen.blit(rotated_image, rect.topleft)
    
def draw_track():
    screen.blit(track_image, (0, 0))

def run_sim(car):
    """Main simulation loop."""
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update the physics world
        world.Step(1.0 / 60.0, 6, 2)
        
        # Clear the screen and redraw
        screen.fill((0, 0, 0))
        draw_track()  # Corrected function call with parentheses
        draw_car(car)
        pygame.display.flip()  # Refresh the screen
        clock.tick(60)

# Future functions for controls, friction, traction, etc.
# def controls
# def friction/traction
# def create_track
# def draw_track      
# start finish   
                             
if __name__ == "__main__":
    car = create_car()
    run_sim(car)  # Start the simulation
