#environment.py

import pygame
import numpy as np

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

gates = [
    {'position': (400, 650), 'orientation': 'vertical1'},  # Example gate
    {'position': (400, 50), 'orientation': 'vertical2'},  # Example gate
    # Add more gates as needed
]

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
            self.speed += 1
        elif action == 1:  # Brake
            self.speed -= 0.2        
        elif action == 2:  # Turn Left
            self.angle += 2
        elif action == 3:  # Turn Right
            self.angle -= 2

        # Update car position
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y += self.speed * np.sin(np.radians(self.angle))


# Function to check if car collides with track boundaries
    def check_collision(self):
        # Logic to check if car has collided with the track boundaries
        distance_from_center = np.sqrt((self.x - center[0])**2 + (self.y - center[1])**2)
        if distance_from_center > outer_radius or distance_from_center < inner_radius:
            #print(f"Xpos: {self.x:.3f}, Ypos: {self.y:.3f}")
            #print(f"Collision detected. Distance from center: {distance_from_center}")
            return True
        return False
    
    def out_bounds(self):
        # Check if car is out of bounds (screen)
        if self.x >= screen_width or self.x < 0 or self.y >= screen_height or self.y < 0:
            #print(f"Xpos: {self.x}, Ypos: {self.y}")
            return True
        return False

    def check_gate_crossing(car, gate):
        # Define a list of gates (checkpoint positions) on the track
        # Car passing the top gate from left to right
        if gate['orientation'] == 'vertical2':
            if gate['position'][0] - 1 <= car.x <= gate['position'][0] + 1:
                if car.x > gate['position'][0] and car.speed > 0:  # Car is moving to the right
                    #print("Car passed top gate from left to right.")
                    return True
        
        # Car passing the bottom gate from right to left
        elif gate['orientation'] == 'vertical1':
            if gate['position'][0] - 1 <= car.x <= gate['position'][0] + 1:
                if car.x < gate['position'][0] and car.speed < 0:  # Car is moving to the left
                    #print("Car passed bottom gate from right to left.")
                    #print(f"Speed {car.speed}")
    
                    return True

        return False  # Return False if no valid crossing
    def forward_motion(car):
        if car.speed >=-0.001:
            print(f"Speed {car.speed}")
    
    def calculate_reward(self):
        reward = 0

        # Reward for forward movement
        if self.speed > 0:
            reward += 2 * self.speed  # Reward for moving forward

        # Penalize for reversing or being slow
        if self.speed < 0:
            reward -= 2  # Penalize for reversing
        elif self.speed < 0.1:
            reward -= 1  # Penalize for moving too slowly

        # Calculate the distance from the center of the track
        distance_from_center = np.sqrt((self.x - center[0])**2 + (self.y - center[1])**2)

        # Reward for staying close to the ideal distance
        ideal_distance = 300
        if 290 <= distance_from_center <= 310:
            reward += 5  # Reward for being near the ideal radius
        else:
            reward -= (abs(distance_from_center - ideal_distance) / 100)  # Penalize for deviation

        # Collision check
        if self.check_collision():
            reward -= 10  # Large penalty for collision

        # Out of bounds check
        if self.out_bounds():
            reward -= 10  # Large penalty for going out of bounds

        # Reward for crossing gates in the correct direction
        for gate in gates:
            if self.check_gate_crossing(gate):
                reward += 5  # Reward for crossing a gate in the correct direction

        # Print debug info for testing
        print(f"Reward: {reward}, X: {self.x}, Y: {self.y}, Speed: {self.speed}, Distance: {distance_from_center}")

        return reward
