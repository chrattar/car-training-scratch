import pygame
from environment import center, outer_radius, inner_radius, WHITE

# Initialize Pygame
BLACK = (0, 0, 0)

def draw_track(screen):
    print(f"Drawing track with center: {center}, outer_radius: {outer_radius}, inner_radius: {inner_radius}")  # Debugging print
    pygame.draw.circle(screen, WHITE, center, outer_radius, 5)  # Outer track
    pygame.draw.circle(screen, WHITE, center, inner_radius, 5)  # Inner track

# Render function to draw the car and track
# Render function to draw the car and track
def render_car(car, screen):
    screen.fill(BLACK)  # Clear the screen
    draw_track(screen)  # Draw the track

    # Draw the car as a rectangle
    car_color = (0, 255, 0)  # Green color for the car
    car_size = (20, 10)  # Car dimensions
    car_rect = pygame.Rect(car.x, car.y, car_size[0], car_size[1])
    pygame.draw.rect(screen, car_color, car_rect)  # Draw the car as a rectangle

    pygame.display.flip()  # Update the display
