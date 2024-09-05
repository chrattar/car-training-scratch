import os
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody, b2Vec2


pygame.init()
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Box2D Car Racing")

#Assets
#File Paths for Assets
current_dir = os.path.dirname(__file__)  # Get the current directory of the script
car_image_path = os.path.join(current_dir, 'car.png')  # Construct full path to 'car.png'
track_image_path = os.path.join(current_dir, 'track.png')  # Construct full path to 'track.png'

# Asset Load from File Paths
car_image = pygame.image.load(car_image_path)
car_image = pygame.transform.scale(car_image, (10, 12))  # Scale the car image as needed
track_image = pygame.image.load(track_image_path)
#World Variables
world = b2World(gravity=(0, 0))


def create_car():
    car_body_def = b2BodyDef()
    car_body_def.type = b2_dynamicBody
    car_body_def.position = (96, 93)  # Set initial position
    car = world.CreateBody(car_body_def)
    car_shape = b2PolygonShape(box=(3.0, 1.8))  # Shape in meters
    car.CreateFixture(shape=car_shape, density=1.0, friction=0.3)
    return car

def draw_car(car):
    position = car.position
    angle = car.angle
    x, y = position[0] * 10, 1080 - position[1] * 10  # Scale Box2D to Pygame (1 meter =a 10 pixels)

    # Rotate the car image according to the Box2D body's angle
    rotated_image = pygame.transform.rotate(car_image, angle * (180.0 / np.pi))  # Rotate image by -90 to align correctly
    rect = rotated_image.get_rect(center=(x, y))
    screen.blit(rotated_image, rect.topleft)
    
def draw_track():
    screen.blit(track_image, (0, 0))

def apply_friction(car):
    """Applies friction to the car to slow it down naturally."""
    # Linear friction (reduces linear velocity)
    velocity = car.linearVelocity
    forward_direction = car.GetWorldVector(localVector=(1, 0))  # Car's forward direction
    forward_speed = b2Vec2.dot(velocity, forward_direction)  # Forward component of the velocity
    lateral_velocity = velocity - forward_direction * forward_speed  # Sideways velocity component

    # Apply lateral friction to reduce sliding
    lateral_friction_impulse = -lateral_velocity * car.mass * 0.5  # Adjust 0.5 for friction strength
    car.ApplyLinearImpulse(lateral_friction_impulse, car.worldCenter, True)

    # Apply linear friction to reduce forward/backward speed
    forward_friction_impulse = -forward_direction * car.mass * 0.1 * forward_speed  # Adjust 0.1 for friction strength
    car.ApplyLinearImpulse(forward_friction_impulse, car.worldCenter, True)

    # Angular friction (reduces angular velocity)
    car.angularVelocity *= 0.95  # Adjust 0.95 for angular friction strength (0 < value < 1)

# Future functions for controls, friction, traction, etc.
def controls(car, max_speed):
    """Applies user controls to the car."""
    keys = pygame.key.get_pressed()

    # Acceleration/Deceleration
    if keys[pygame.K_UP]:
        # Apply forward force in the direction the car is facing
        forward_direction = car.GetWorldVector(localVector=(1, 1))  # Local Y-axis as forward
        forward_force = 1000 * forward_direction  # Scale force magnitude as needed
        car.ApplyForceToCenter(forward_force, True)
        print(f"Forward Force: {forward_force}")

    if keys[pygame.K_DOWN]:
        # Apply backward force in the opposite direction the car is facing
        backward_direction = car.GetWorldVector(localVector=(1, -1))  # Local negative Y-axis as backward
        backward_force = 500 * backward_direction  # Reduced force for braking
        car.ApplyForceToCenter(backward_force, True)
        print(f"Backward Force: {backward_force}")

    # Steering controls
    if keys[pygame.K_LEFT]:
        car.angularVelocity = 5.0  # Adjust turning speed
    elif keys[pygame.K_RIGHT]:
        car.angularVelocity = -5.0  # Adjust turning speed
    else:
        car.angularVelocity = 0  # Stop rotating when no keys are pressed

    # Cap speed at max_speed (10 m/s)
    velocity = car.linearVelocity.length
    if velocity > max_speed:
        car.linearVelocity *= max_speed / velocity
    print(f"Car velocity: {car.linearVelocity.length:.2f} m/s, Position: {car.position}")

def check_boundaries(car):
    """Keeps the car within the screen boundaries."""
    # Convert Box2D position to Pygame coordinates
    x, y = car.position[0] * 10, 1080 - car.position[1] * 10  # Scale from meters to pixels

    # Screen boundaries in Pygame coordinates
    min_x, max_x = 0, 1920
    min_y, max_y = 0, 1080

    # Check boundaries for X-axis
    if x < min_x:
        car.linearVelocity.x = 0  # Stop car movement in the X direction
        car.angularVelocity = 0  # Stop car rotation
        car.position.x = min_x / 10  # Reset to the left boundary
    elif x > max_x:
        car.linearVelocity.x = 0  # Stop car movement in the X direction
        car.angularVelocity = 0  # Stop car rotation
        car.position.x = max_x / 10  # Reset to the right boundary

    # Check boundaries for Y-axis
    if y < min_y:
        car.linearVelocity.y = 0  # Stop car movement in the Y direction
        car.angularVelocity = 0  # Stop car rotation
        car.position.y = (1080 - min_y) / 10  # Reset to the bottom boundary
    elif y > max_y:
        car.linearVelocity.y = 0  # Stop car movement in the Y direction
        car.angularVelocity = 0  # Stop car rotation
        car.position.y = (1080 - max_y) / 10  # Reset to the top boundary

def draw_start_finish_line():
    """Draws the start/finish line on the track."""
    line_color = (255, 144, 44)  # Red color
    line_start_pos = (962, 100)  # Starting position of the line in Pygame coordinates
    line_end_pos = (962, 198)  # Ending position of the line in Pygame coordinates
    pygame.draw.line(screen, line_color, line_start_pos, line_end_pos, 5)  # Line width of 5 pixels


def run_sim(car):
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
      
        controls(car, max_speed=10.0)  # Max speed set to 10 m/s
        apply_friction(car)
        check_boundaries(car)
        # Update the physics world
        world.Step(1.0 / 60.0, 6, 2)#Update Physics
        
        # Clear the screen and redraw
        screen.fill((0, 0, 0))
        draw_track()  
        draw_car(car)
        draw_start_finish_line()
        pygame.display.flip()  # Refresh the screen
        clock.tick(60)
        

                             
if __name__ == "__main__":
    car = create_car()
    run_sim(car)  # Start the simulation
