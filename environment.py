#environment.py

import pygame
import numpy as np
import math


#CONFIG
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
outer_radius = 350
inner_radius = 250
screen_width, screen_height = 800, 800
center = (screen_width // 2, screen_height // 2)
radius = 300

gates = []
for i in range(8):
    base_angle = -1 * (3 * math.pi / 2 - i * math.pi / 4) % (2 * math.pi)  # Start at bottom and go clockwise
    angle = (-1 * base_angle) % (2 * math.pi)  
    x = center[0] + int(radius * math.sin(angle))
    y = center[1] + int(radius * math.cos(angle))
    gates.append({
        'position': (x, y),
        'orientation': 'radial',
        'number': i,
        'angle': angle
    })


#CAR VARIABLES
car_size = (20, 10)  # width, height

class Car:
    def __init__(self):
        self.x = center[0]  # Start at the center
        self.prev_x = self.x
        self.y = center[1] + inner_radius + 20  # Position outside the inner circle
        self.angle = 0  # Start angle
        self.speed = 0  # Start speed
        self.last_gate = 0
        self.reset()
            
    def reset(self):
        # Reset car to initial state
        self.x = center[0]
        self.y = center[1] + inner_radius + 20
        self.angle = 0  # Reset angle
        self.speed = 0  # Reset speed

    def move(self, action):#TURNING AND FWD
        self.prev_x = self.x
        self.prev_y = self.y
        if action == 0:  
            self.speed += 3
        elif action == 1: 
            self.speed -= 0.1
            if self.speed < 0:
                self.speed = 0  
        elif action == 2: 
            self.angle += 6
        elif action == 3: 
            self.angle -= 6
        self.x += self.speed * np.cos(np.radians(self.angle)) 
        self.y -= self.speed * np.sin(np.radians(self.angle))  

    def check_collision(self):

        distance_from_center = np.sqrt((self.x - center[0])**2 + (self.y - center[1])**2)
        if distance_from_center > outer_radius or distance_from_center < inner_radius:
            return True
        return False
    
    def out_bounds(self):

        if self.x >= screen_width or self.x < 0 or self.y >= screen_height or self.y < 0:
            return True
        return False

    def check_gate_crossing(self, gate):
        gate_x, gate_y = gate['position']
        gate_vector = (gate_x - center[0], gate_y - center[1])
        prev_car_vector = (self.prev_x - center[0], self.prev_y - center[1])
        car_vector = (self.x - center[0], self.y - center[1])
        
        prev_cross_product = gate_vector[0] * prev_car_vector[1] - gate_vector[1] * prev_car_vector[0]
        cross_product = gate_vector[0] * car_vector[1] - gate_vector[1] * car_vector[0]

        distance_to_gate = math.sqrt((self.x - gate_x)**2 + (self.y - gate_y)**2)#How far from gate
        
        if distance_to_gate < 20 and prev_cross_product <= 0 and cross_product > 0:
            return True
        return False

    def forward_motion(car):
        if car.speed >=-0.001:
            print(f"Speed {car.speed}")
    
    def calculate_reward(self):
        reward = 0
        if self.speed > 0:
            reward += 0.2 * self.speed
        distance_from_center = math.sqrt((self.x - center[0])**2 + (self.y - center[1])**2)
        if abs(distance_from_center - radius) <= 20:
            reward += 2
        else:
            reward -= 1
        next_gate = (self.last_gate + 1) % 8
        if self.check_gate_crossing(gates[next_gate]):
            reward += 50
            self.last_gate = next_gate
            print(f"Crossed gate {next_gate}")
        
        if self.check_collision() or self.out_bounds():
            reward -= 100

        return reward

       
