import os
import pygame
import numpy as np
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody, b2Vec2

pygame.init()
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Box2D Car Racing")

# Assets
current_dir = os.path.dirname(__file__)  # Get the current directory of the script
car_image_path = os.path.join(current_dir, 'car.png')  # Construct full path to 'car.png'
track_image_path = os.path.join(current_dir, 'track.png')  # Construct full path to 'track.png'

# Load and scale assets
car_image = pygame.image.load(car_image_path)
car_image = pygame.transform.scale(car_image, (10, 12))
track_image = pygame.image.load(track_image_path)

# World Variables
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
    x, y = position[0] * 10, 1080 - position[1] * 10  # Scale Box2D to Pygame (1 meter = 10 pixels)

    # Rotate the car image according to the Box2D body's angle
    rotated_image = pygame.transform.rotate(car_image, angle * (180.0 / np.pi))
    rect = rotated_image.get_rect(center=(x, y))
    screen.blit(rotated_image, rect.topleft)
    
def draw_track():
    screen.blit(track_image, (0, 0))

def apply_friction(car):
    """Applies friction to the car to slow it down naturally."""
    velocity = car.linearVelocity
    forward_direction = car.GetWorldVector(localVector=(1, 0))  # Car's forward direction
    forward_speed = b2Vec2.dot(velocity, forward_direction)  # Forward component of the velocity
    lateral_velocity = velocity - forward_direction * forward_speed  # Sideways velocity component

    # Apply lateral friction to reduce sliding
    lateral_friction_impulse = -lateral_velocity * car.mass * 0.5
    car.ApplyLinearImpulse(lateral_friction_impulse, car.worldCenter, True)

    # Apply linear friction to reduce forward/backward speed
    forward_friction_impulse = -forward_direction * car.mass * 0.1 * forward_speed
    car.ApplyLinearImpulse(forward_friction_impulse, car.worldCenter, True)

    # Angular friction (reduces angular velocity)
    car.angularVelocity *= 0.95

def controls(car, max_speed):
    """Applies user controls to the car."""
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        forward_direction = car.GetWorldVector(localVector=(1, 1))
        forward_force = 10000 * forward_direction
        car.ApplyForceToCenter(forward_force, True)
        print(f"Forward Force: {forward_force}")

    if keys[pygame.K_DOWN]:
        backward_direction = car.GetWorldVector(localVector=(1, -1))
        backward_force = 50000 * backward_direction
        car.ApplyForceToCenter(backward_force, True)
        print(f"Backward Force: {backward_force}")

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
    print(f"Car velocity: {car.linearVelocity.length:.2f} m/s, Position: {car.position}")

def check_boundaries(car):
    """Keeps the car within the screen boundaries."""
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

def draw_start_finish_line():
    """Draws the start/finish line on the track."""
    line_color = (255, 23, 44)  # Red color
    line_start_pos = (962, 100)
    line_end_pos = (962, 198)
    pygame.draw.line(screen, line_color, line_start_pos, line_end_pos, 5)

gates = [
    {'x': 1295, 'y': 100, 'direction': 'gate1_v'},
    {'x': 1628, 'y': 475, 'direction': 'gate2_h'},
    {'x': 1295, 'y': 804, 'direction': 'gate3_v'},
    {'x': 695, 'y': 804, 'direction': 'gate4_v'},
    {'x': 160, 'y': 475, 'direction': 'gate5_h'},
    {'x': 695, 'y': 100, 'direction': 'gate6_v'}
]

def draw_gates():
    for gate in gates:
        gate_color = (0, 255, 0)
        if 'v' in gate['direction']:
            pygame.draw.line(screen, gate_color, (gate['x'], gate['y']), (gate['x'], gate['y'] + 100), 5)
        elif 'h' in gate['direction']:
            pygame.draw.line(screen, gate_color, (gate['x'], gate['y']), (gate['x'] + 188, gate['y']), 5)

def check_gate_crossing(car, gates, lap_counter):
    """Checks if the car has crossed a gate and updates the lap counter."""
    x, y = car.position[0] * 10, 1080 - car.position[1] * 10  # Convert Box2D position to Pygame coordinates
    current_gate = gates[lap_counter['current_gate']]  # Get the current gate to check

    # Get gate position and direction
    gate_x, gate_y = current_gate['x'], current_gate['y']
    direction = current_gate['direction']

    # Check if the car has crossed the current gate
    crossed = False
    if 'v' in direction:  # Vertical gate
        if abs(x - gate_x) < 5 and gate_y <= y <= (gate_y + 100):
            crossed = True
    elif 'h' in direction:  # Horizontal gate
        if abs(y - gate_y) < 5 and gate_x <= x <= (gate_x + 40):
            crossed = True

    # If crossed, move to the next gate
    if crossed:
        lap_counter['current_gate'] += 1
        print(f"Crossed gate {lap_counter['current_gate']}")

        # If all gates crossed, increment lap counter
        if lap_counter['current_gate'] == len(gates):
            lap_counter['lap_count'] += 1
            lap_counter['current_gate'] = 0  # Reset to the first gate
            print(f"Lap completed! Total Laps: {lap_counter['lap_count']}")

def draw_lap_count(screen, lap_counter):
    """Displays the lap count on the screen."""
    font = pygame.font.SysFont(None, 36)
    lap_text = font.render(f"Laps: {lap_counter['lap_count']}", True, (255, 255, 255))  # White color
    screen.blit(lap_text, (10, 10))

def run_sim(car):
    # Initialize lap counter
    lap_counter = {'current_gate': 0, 'lap_count': 0}
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
      
        controls(car, max_speed=100.0)
        apply_friction(car)
        check_boundaries(car)
        check_gate_crossing(car, gates, lap_counter)
        world.Step(1.0 / 60.0, 6, 2)
        
        screen.fill((0, 0, 0))
        draw_track()
        draw_car(car)
        draw_start_finish_line()
        draw_gates()
        draw_lap_count(screen, lap_counter)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    car = create_car()
    run_sim(car)
