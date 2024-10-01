import pygame
from environment import center, outer_radius, inner_radius, WHITE

# Initialize Pygame
BLACK = (0, 0, 0)

def draw_track(screen):
    pygame.draw.circle(screen, WHITE, center, outer_radius, 5)  # Outer track
    pygame.draw.circle(screen, WHITE, center, inner_radius, 5)  # Inner track

# Define a list of gates (checkpoint positions) on the track
gates = [
    {'position': (400, 650), 'orientation': 'vertical1'},  # Example gate
    {'position': (400, 50), 'orientation': 'vertical2'},  # Example gate
    # Add more gates as needed
]

def draw_gates(screen):
    for gate in gates:
        if gate['orientation'] == 'vertical1':
            pygame.draw.line(screen, (255, 0, 0), gate['position'], (gate['position'][0], gate['position'][1] + 95), 5)  # Vertical line gate
        elif gate['orientation'] == 'vertical2':
            pygame.draw.line(screen, (255, 0, 0), gate['position'], (gate['position'][0], gate['position'][1] + 95), 5)  # Vertical line gate 
        elif gate['orientation'] == 'horizontal':
            pygame.draw.line(screen, (255, 0, 0), gate['position'], (gate['position'][0] + 50, gate['position'][1]), 5)  # Horizontal line gate
# Render function to draw the car and track

def render_car(car, screen):
    screen.fill(BLACK)  # Clear the screen
    draw_track(screen)  # Draw the track
    draw_gates(screen)
    # Draw the car as a rectangle
    car_color = (0, 255, 0)  # Green color for the car
    car_size = (20, 10)  # Car dimensions
    car_rect = pygame.Rect(car.x, car.y, car_size[0], car_size[1])
    pygame.draw.rect(screen, car_color, car_rect)  # Draw the car as a rectangle
    

    pygame.display.flip()  # Update the display
