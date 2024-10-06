import pygame
from environment import center, outer_radius, inner_radius, WHITE, gates, radius, angle
import math

# Initialize Pygame
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
screen_width, screen_height = 800, 800
center = (screen_width // 2, screen_height // 2)
radius = 300  # Adjust this value based on your track size

def draw_track(screen):
    pygame.draw.circle(screen, WHITE, center, outer_radius, 5)  # Outer track
    pygame.draw.circle(screen, WHITE, center, inner_radius, 5)  # Inner track

def draw_gates(screen):
    for gate in gates:
        gate_color = GREEN if gate['number'] == 0 else YELLOW
        pygame.draw.circle(screen, gate_color, gate['position'], 5)
        
        font = pygame.font.Font(None, 24)
        text = font.render(str(gate['number']), True, gate_color)
        
        # Calculate text position outside the track
        text_distance = 30  # Distance from the gate circle
        
        # If 'angle' is not in the gate dictionary, calculate it
        if 'angle' not in gate:
            x, y = gate['position']
            angle = math.atan2(y - center[1], x - center[0])
        else:
            angle = gate['angle']
        
        text_x = center[0] + int((radius + text_distance) * math.cos(angle))
        text_y = center[1] + int((radius + text_distance) * math.sin(angle))
        
        text_rect = text.get_rect(center=(text_x, text_y))
        screen.blit(text, text_rect)
        
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
