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
        self.prev_x = self.x
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
        self.prev = self.x
        if action == 0:  # Accelerate forward
            self.speed += 3
        elif action == 1:  # Brake
            self.speed -= 0.1
            if self.speed < 0:
                self.speed = 0  # Prevent reverse through braking
        elif action == 2:  # Turn Left
            self.angle += 6
        elif action == 3:  # Turn Right
            self.angle -= 6

        # Update car position based on speed and angle
        self.x += self.speed * np.cos(np.radians(self.angle))  # Forward movement based on angle
        self.y -= self.speed * np.sin(np.radians(self.angle))  # Adjust y direction to fit screen coordinates



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

    def check_gate_crossing(self, gate):
        tolerance = 10  # Increased tolerance
        
        if gate['orientation'] == 'vertical2':
            if abs(self.x - gate['position'][0]) <= tolerance:
                if self.prev_x <= gate['position'][0] and self.x > gate['position'][0]:
                    return True
        elif gate['orientation'] == 'vertical1':
            if abs(self.x - gate['position'][0]) <= tolerance:
                if self.prev_x >= gate['position'][0] and self.x < gate['position'][0]:
                    return True
        
        return False

    def forward_motion(car):
        if car.speed >=-0.001:
            print(f"Speed {car.speed}")
    
    def calculate_reward(self):
        reward = 0

        # Calculate distance from center
        distance_from_center = np.sqrt((self.x - center[0])**2 + (self.y - center[1])**2)

        # Reward for staying within the track
        if inner_radius < distance_from_center < outer_radius:
            reward += 1
        else:
            reward -= 10  # Significant penalty for going out of bounds

        # Reward for moving
        if self.speed > 0:
            reward += 0.1 * self.speed

        # Reward for moving in a circular path
        ideal_angle = (np.degrees(np.arctan2(-(self.y - center[1]), self.x - center[0])) + 90) % 360
        angle_diff = abs((self.angle - ideal_angle + 180) % 360 - 180)
        if angle_diff < 20:  # If the car is moving tangent to the circle (with some tolerance)
            reward += 1
        print(f"Reward: {reward}, X: {self.x:.4f}, Y: {self.y:.4f}, Speed: {self.speed:.4f}, Distance: {distance_from_center:.4f}")
        return reward

        # Print debug info for testing
      

       
