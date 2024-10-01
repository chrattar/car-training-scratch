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

def render_car(self, screen):
    # Define car size
    car_width = 20
    car_height = 10
    
    # Create the car's body as a surface
    car_body = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    
    # Fill the car's body with a base color (e.g., green)
    car_body.fill((0, 255, 0))
    
    # Draw the front of the car as a red rectangle
    front_rect = pygame.Rect(0, 0, car_width // 2, car_height)  # Define the front half
    pygame.draw.rect(car_body, (255, 0, 0), front_rect)  # Color the front red
    
    # Rotate the car around its center based on its angle
    rotated_body = pygame.transform.rotate(car_body, -self.angle)
    
    # Get the new rectangle (for positioning) and blit the car to the screen
    rotated_rect = rotated_body.get_rect(center=(self.x, self.y))
    screen.blit(rotated_body, rotated_rect.topleft)


    pygame.display.flip()  # Update the display
