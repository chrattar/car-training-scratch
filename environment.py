# environment.py

import numpy as np
from car_data import create_car, check_lap_completion, car_collision, car_out_of_bounds, check_boundaries, draw_car, apply_friction
from track import draw_track, draw_start_finish_line
from config import world, screen, car_image
from utils import initialize_lap_counter
import pygame
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody, b2Vec2


class CustomCarEnv:
    def __init__(self):
        """
        Initializes the custom car environment.
        """
        self.car = None
        self.lap_counter = None
        self.prev_distance_to_goal = None  # Store the previous distance to compute progress
        self.reset()

    def reset(self):
        """Resets the environment to an initial state and returns the initial observation.

        Returns:
            initial_state: The initial state of the environment."""
        # Reset the car to its initial position
        self.car = create_car()  # Set up the car object in its initial state
        # Reset environment variables such as laps, position, speed, etc.
        self.lap_counter = initialize_lap_counter()  # Reset lap counter to zero
        self.prev_distance_to_goal = self.compute_distance_to_goal()  # Initialize the distance to the goal
        
        # Clear screen or set up initial screen state
        screen.fill((0, 0, 0))  # Clear the screen with a black background
        draw_track(screen)  # Redraw the track
        draw_car(self.car, screen)
        draw_start_finish_line(screen)  # Redraw the start/finish line
        
        # Return the initial state, which might be car's position, velocity, etc.
        initial_state = [self.car.position.x, self.car.position.y, 0, 0]  # Example state (x, y, velocity, lap_count)
        return initial_state

    def compute_distance_to_goal(self):
        """Computes the distance from the car to the goal (finish line or next checkpoint)."""
        # Assume the goal is the origin (0, 0) for simplicity; adjust as needed for your environment
        goal_position = np.array([0, 0])
        car_position = np.array([self.car.position.x, self.car.position.y])
        distance = np.linalg.norm(car_position - goal_position)
        return distance

    def step(self, action):
        """Perform one step in the environment based on the chosen action.
        Args:
            action: The action selected by the agent (e.g., 0: forward, 1: left, 2: right).

        Returns:
            next_state: The next state after performing the action.
            reward: The reward obtained from the action.
            done: Boolean indicating if the episode is complete.
            info: Additional information (optional, can be used for debugging)."""
        # Define actions
        FORWARD, LEFT, RIGHT = 0, 1, 2

        # Apply action using Box2D methods directly
        if action == FORWARD:
            self.car.ApplyForceToCenter((8000.0, 0.0), True)  # Apply a stronger force to move the car forward
        elif action == LEFT:
            self.car.ApplyTorque(50.0, True)  # Apply a torque to turn left
        elif action == RIGHT:
            self.car.ApplyTorque(-50.0, True)  # Apply a torque to turn right

        # Apply friction to control unwanted sliding and jiggling
        apply_friction(self.car)

        # Step the simulation forward
        world.Step(1.0 / 60.0, 8, 3)  # Update physics with Box2D (increase velocity/position iterations)

        # Calculate reward
        reward = 0
        done = False
        
        # Reward for moving forward
        if action == FORWARD:
            reward += 2  # Small positive reward for each forward step
            done = False

        # Penalize for going off-track
        if not check_boundaries(self.car):
            reward -= 100  # Large negative reward for going off-track
            #done = True  # Only end the episode if the car is significantly off-track
            done = False

        # Reward for completing a lap
        if check_lap_completion(self.car, self.lap_counter):
            reward += 500  # Large reward for completing a lap
            #done = True
            done = False

        # Distance-based reward for moving toward the goal
        distance_to_goal = self.compute_distance_to_goal()
        if distance_to_goal < self.prev_distance_to_goal:
            reward += 1  # Reward for reducing distance to goal
        self.prev_distance_to_goal = distance_to_goal

        # Penalize for staying still or moving too slowly
        if np.linalg.norm((self.car.linearVelocity.x, self.car.linearVelocity.y)) < 0.1:
            reward -= 5  # Moderate negative reward for low speed
            done = False

        # Compute the next state
        next_state = [self.car.position.x, self.car.position.y, self.car.linearVelocity.x, self.car.linearVelocity.y]

        # Check if the car is truly out of bounds
        if car_out_of_bounds(self.car):
            reward -= 50  # Penalty for being out of bounds
            done = False  # Allow some buffer before ending the episode

        # Check if the car has collided
        if car_collision(self.car):
           done = False  # End episode if the car crashes

        # Return the next state, reward, done flag, and additional info
        return np.array(next_state), reward, done, {}
