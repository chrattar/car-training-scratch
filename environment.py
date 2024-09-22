#environment.py

import pygame
import numpy as np

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Track dimensions
outer_radius = 350
inner_radius = 250
screen_width, screen_height = 800, 800
center = (screen_width // 2, screen_height // 2)

# Car parameters
car_size = (20, 10)  # width, height

class Car:
    def __init__(self):
        self.x = center[0]  # Start at the center
        self.y = center[1] + inner_radius + 20  # Position outside the inner circle
        self.angle = 0  # Start angle
        self.speed = 0  # Start speed
        self.reset()
            
    def reset(self):
        # Reset car to initial state
        self.x = center[0]
        self.y = center[1] + inner_radius + 20
        self.angle = 0  # Reset angle
        self.speed = 0  # Reset speed

    def move(self, action):
        if action == 0:  # Accelerate
            self.speed += 0.1
        elif action == 1:  # Brake
            self.speed -= 0.1
        elif action == 2:  # Turn Left
            self.angle += 5
        elif action == 3:  # Turn Right
            self.angle -= 5

        # Update car position
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y -= self.speed * np.sin(np.radians(self.angle))


# Function to check if car collides with track boundaries
    def check_collision(self):
        # Logic to check if car has collided with the track boundaries
        distance_from_center = np.sqrt((self.x - center[0])**2 + (self.y - center[1])**2)
        if distance_from_center > outer_radius or distance_from_center < inner_radius:
            print(f"Collision detected. Distance from center: {distance_from_center}")
            return True
        return False
    
    def out_bounds(self):
        if self.x >= screen_width or self.x < 0 or self.y >= screen_height or self.y < 0:
            print(f"Out of bounds: Car has exited the screen at X: {self.x:.2f}, Y: {self.y:.2f}")
            return True
        return False
    
    def calculate_reward(car):
    # Positive reward for forward movement
        reward = car.speed * np.cos(np.radians(car.angle))  # Reward for moving forward

        # Penalize or reward based on how close the car is to the ideal distance (300 px)
        ideal_distance = 300
        outer_radius = 350
        inner_radius = 250
        center = (400, 400)
        
        # Calculate the distance from the center of the track
        distance_from_center = np.sqrt((car.x - center[0])**2 + (car.y - center[1])**2)
        
        # Penalize for large deviations from the ideal distance
        if 290 <= distance_from_center <= 310:
            reward += 10  # High reward for being close to 300px
        else:
            reward -= 2  # Small penalty for deviation

        return reward

