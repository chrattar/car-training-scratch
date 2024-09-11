#car_data.py
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody, b2Vec2
from config import world, car_image, screen
import pygame
from utils import line_end_pos, line_start_pos
import numpy as np



def create_car():
    car_body_def = b2BodyDef()
    car_body_def.type = b2_dynamicBody
    car_body_def.position = (96, 93)  # Set initial position
    car_body_def.allowSleep = True
    car = world.CreateBody(car_body_def)
    car_shape = b2PolygonShape(box=(3.0, 3.0))  # Shape in meters
    car.CreateFixture(shape=car_shape, density=1.0, friction=0.3)
    return car

def draw_car(car, screen):
    position = car.position
    angle = car.angle
    x, y = position[0] * 10, 1080 - position[1] * 10  # Scale Box2D to Pygame (1 meter = 10 pixels)

    # Rotate the car image according to the Box2D body's angle
    rotated_image = pygame.transform.rotate(car_image, (angle * (180.0 / np.pi)) )
    rect = rotated_image.get_rect(center=(x, y))
    screen.blit(rotated_image, rect.topleft)
    
def apply_friction(car):
    velocity = car.linearVelocity
    angular_velocity = car.angularVelocity
    
    # Thresholds to prevent jiggling
    velocity_threshold = 0.1  # Threshold for linear velocity
    angular_velocity_threshold = 0.1  # Threshold for angular velocity

    # Calculate the car's forward direction
    forward_direction = car.GetWorldVector(localVector=(1, 0))  # Car's forward direction
    forward_speed = b2Vec2.dot(velocity, forward_direction)  # Forward component of the velocity

    # Calculate lateral velocity to reduce sliding
    lateral_direction = car.GetWorldVector(localVector=(0, 1))  # Car's lateral direction
    lateral_speed = b2Vec2.dot(velocity, lateral_direction)  # Lateral component of the velocity
    lateral_velocity = lateral_speed * lateral_direction

    # Apply lateral friction to reduce sliding
    lateral_friction_impulse = -lateral_velocity * car.mass * 2.0
    car.ApplyLinearImpulse(lateral_friction_impulse, car.worldCenter, True)

    # Apply linear friction to reduce forward/backward speed gradually
    forward_friction_impulse = -forward_direction * car.mass * 0.1 * forward_speed
    car.ApplyLinearImpulse(forward_friction_impulse, car.worldCenter, True)

    # Apply angular friction to reduce rotational velocity gradually
    car.angularVelocity *= 0.8

    # Check if the velocities are below the threshold; if so, set them to zero
    if velocity.length < velocity_threshold:
        car.linearVelocity.SetZero()

    if abs(angular_velocity) < angular_velocity_threshold:
        car.angularVelocity = 0

def controls(car, max_speed):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        forward_direction = car.GetWorldVector(localVector=(1, 0))
        forward_force = 10000 * forward_direction
        car.ApplyForceToCenter(forward_force, True)
       # print(f"Forward Force: {forward_force}")

    if keys[pygame.K_DOWN]:
        backward_direction = car.GetWorldVector(localVector=(-1, 0))
        backward_force = 5000 * backward_direction
        car.ApplyForceToCenter(backward_force, True)
        #print(f"Backward Force: {backward_force}")

    if keys[pygame.K_LEFT]:
        car.angularVelocity = 5.0
    elif keys[pygame.K_RIGHT]:
        car.angularVelocity = -5.0
    else:
        car.angularVelocity = 0

    # Cap speed at max_speed
    velocity = car.linearVelocity.length
    if velocity > max_speed:
        car.linearVelocity *= max_speed / velocity
    #print(f"Car velocity: {car.linearVelocity.length:.2f} m/s, Position: {car.position}")

def check_boundaries(car):
    x, y = car.position[0] * 10, 1080 - car.position[1] * 10

    min_x, max_x = 0, 1920
    min_y, max_y = 0, 1080

    if x < min_x:
        car.linearVelocity.x = 0
        car.angularVelocity = 0
        car.position.x = min_x / 10
    elif x > max_x:
        car.linearVelocity.x = 0
        car.angularVelocity = 0
        car.position.x = max_x / 10

    if y < min_y:
        car.linearVelocity.y = 0
        car.angularVelocity = 0
        car.position.y = (1080 - min_y) / 10
    elif y > max_y:
        car.linearVelocity.y = 0
        car.angularVelocity = 0
        car.position.y = (1080 - max_y) / 10

def check_lap_completion(car, lap_counter):
    x, y = car.position[0] * 10, 1080 - car.position[1] * 10

    # Check if the car crosses the start/finish line
    if line_start_pos[0] - 5 <= x <= line_start_pos[0] + 5 and line_start_pos[1] <= y <= line_end_pos[1]:
        # Check if the car is moving in the correct direction
        if car.linearVelocity.x > 0 and not lap_counter['crossing']:  #Lap needs to check in increasing x direction
            lap_counter['lap_count'] += 1
            lap_counter['crossing'] = True  #Cross Flag
            #print(f"Lap completed! Total Laps: {lap_counter['lap_count']}")
    else:
        # Reset the crossing flag when the car is away from the line
        lap_counter['crossing'] = False
        
def car_out_of_bounds(car):
    """
    Checks if the car is out of the track boundaries.

    Args:
        car: The car object representing the agent in the environment.

    Returns:
        Boolean: True if the car is out of bounds, False otherwise.
    """
    # Define the screen or track boundaries
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # Adjust to your environment's dimensions

    # Get the car's current position
    car_x, car_y = car.position.x, car.position.y

    # Check if the car is within the screen boundaries
    if car_x < 0 or car_x > SCREEN_WIDTH or car_y < 0 or car_y > SCREEN_HEIGHT:
        return True  # The car is out of bounds
    return False

def car_collision(car):
    """
    Checks if the car has collided with any obstacles.

    Args:
        car: The car object representing the agent in the environment.

    Returns:
        Boolean: True if the car has collided, False otherwise.
    """
    # Example: Check if car has collided with the track boundaries or obstacles
    # This requires a specific collision detection mechanism based on your environment.
    
    # Placeholder for collision detection logic
    # If using Box2D, you might use car.contactList or similar to check for collisions
    contact_list = car.contacts  # Example using Box2D contact list
    if contact_list:
        return True  # A collision has occurred

    return False

