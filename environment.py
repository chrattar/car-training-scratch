import math
from Box2D import b2World, b2PolygonShape, b2BodyDef, b2_dynamicBody, b2CircleShape
from render import draw_environment
from config import TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS, MAX_SPEED, SCREEN_WIDTH, SCREEN_HEIGHT
import numpy as np

class Environment:
    def __init__(self):
        self.world = b2World(gravity=(0, 0), doSleep=True)  # Initialize Box2D world

        # Create collision boundaries (edges of the circular track)
        self.track_boundaries = self.create_track_boundaries()

        # Create the car
        self.car = self.create_car()

    def create_track_boundaries(self):
        # Center and radius for the track
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        radius_outer = 400 / 10  # Convert to meters (Box2D units)
        radius_inner = 170 / 10  # Inner radius for the track

        track_bodies = []

        # Create outer boundary as an octagon approximation of the circle
        outer_vertices = [
            (radius_outer * math.cos(angle) + center_x / 10, radius_outer * math.sin(angle) + center_y / 10)
            for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False)
        ]

        outer_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(vertices=outer_vertices)
        )
        track_bodies.append(outer_body)

        # Create inner boundary as an octagon approximation of the inner circle
        inner_vertices = [
            (radius_inner * math.cos(angle) + center_x / 10, radius_inner * math.sin(angle) + center_y / 10)
            for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False)
        ]

        inner_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(vertices=inner_vertices)
        )
        track_bodies.append(inner_body)

        return track_bodies
  
    def create_car(self):
        # Center of the track in pixels
        center_x = SCREEN_WIDTH / 2  # 960 pixels
        center_y = SCREEN_HEIGHT / 2  # 540 pixels

        # Desired initial position: inside the bottom of the circular track
        initial_x = center_x / 10  # Convert x position to meters
        initial_y = (center_y + 350) / 10  # Adjust y position to be 50 pixels inside the outer radius (400 px)

        # Create the car at the specified position
        car_body_def = b2BodyDef(type=b2_dynamicBody, position=(initial_x, initial_y), allowSleep=True)
        car = self.world.CreateBody(car_body_def)
        car_shape = b2PolygonShape(box=(3.0, 1.5))  # Car size in meters
        car.CreateFixture(shape=car_shape, density=1.0, friction=0.3)
        return car

    def reset(self):
        # Center of the track in pixels
        center_x = SCREEN_WIDTH / 2  # 960 pixels
        center_y = SCREEN_HEIGHT / 2  # 540 pixels

        # Reset car's position to just inside the bottom edge of the circular track
        initial_x = center_x / 10  # Convert x position to meters
        initial_y = (center_y + 350) / 10  # Adjust y position to be 50 pixels inside the outer radius (400 px)

        self.car.position = (initial_x, initial_y)  # Set to the initial position
        self.car.linearVelocity = (0, 0)
        self.car.angularVelocity = 0
        return self.get_state()  # Return initial state representation



    def step(self, action):
        self.apply_action(action)  # Apply the action to the car
        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)  # Advance simulation

        reward, done = self.calculate_reward_and_done()  # Compute reward and check if the episode is done
        next_state = self.get_state()  # Get the next state
        return next_state, reward, done

    def apply_action(self, action):
        if action == 0:  # Accelerate forward
            forward_force = self.car.GetWorldVector((1, 0)) * 1000
            self.car.ApplyForceToCenter(forward_force, True)
        elif action == 1:  # Decelerate / brake
            backward_force = self.car.GetWorldVector((-1, 0)) * 500
            self.car.ApplyForceToCenter(backward_force, True)
        elif action == 2:  # Turn left
            self.car.ApplyTorque(50.0, True)
        elif action == 3:  # Turn right
            self.car.ApplyTorque(-50.0, True)
    def calculate_reward_and_done(self):
        # Initialize reward
        reward = 0

        # Car's position in pixels
        car_x, car_y = self.car.position[0] * 10, SCREEN_HEIGHT - self.car.position[1] * 10

        # Track center and radii
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        outer_radius = 400
        inner_radius = 170

        # Calculate distance from the car to the center of the track
        distance_to_center = ((car_x - center_x) ** 2 + (car_y - center_y) ** 2) ** 0.5

        # Check if the car is within the track boundaries
        if inner_radius <= distance_to_center <= outer_radius:
            # Positive reward for staying on the track
            reward += 1

            # Calculate forward velocity along the desired direction (tangential to the circular track)
            car_velocity = self.car.linearVelocity
            car_direction = math.atan2(car_y - center_y, car_x - center_x)  # Angle from center to car
            desired_direction = car_direction + math.pi / 2  # Tangential direction

            # Dot product to find alignment with the tangential direction
            tangential_velocity = car_velocity[0] * math.cos(desired_direction) + car_velocity[1] * math.sin(desired_direction)

            # Reward based on forward movement along the circular path
            if tangential_velocity > 0:
                reward += tangential_velocity * 0.1  # Scale reward by velocity along the desired direction
            else:
                reward -= abs(tangential_velocity) * 0.1  # Penalize backward movement

            done = False  # The car is still within the track boundaries

        else:
            # Negative reward for leaving the track
            reward -= 100
            done = True  # The car has left the track, episode is done

        return reward, done


    def get_state(self):
        # Get the car's position, velocity, and angle
        car_position = self.car.position
        car_velocity = self.car.linearVelocity
        car_angle = self.car.angle

        # Create a state vector
        state = [
            float(car_position[0]), float(car_position[1]),  # X, Y position
            float(car_velocity[0]), float(car_velocity[1]),  # X, Y velocity
            float(car_angle)  # Orientation
        ]
        return state
