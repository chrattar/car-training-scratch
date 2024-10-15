import pygame
import math
from environment import center, outer_radius, inner_radius, gates
import os

def draw_track(screen):
    pygame.draw.circle(screen, (255, 255, 255), center, outer_radius, 2)
    pygame.draw.circle(screen, (255, 255, 255), center, inner_radius, 2)

def draw_gates(screen):
    for gate in gates:
        gate_color = (0, 255, 0) if gate['number'] == 0 else (255, 255, 0)
        pygame.draw.line(screen, gate_color, gate['inner_point'], gate['outer_point'], 2)

def render_rays(car, screen, num_rays, ray_length):
    for i in range(num_rays):
        angle = math.radians(car.angle + (i - num_rays // 2) * (360 / num_rays))
        end_x = car.x + ray_length * math.cos(angle)
        end_y = car.y - ray_length * math.sin(angle)
        pygame.draw.line(screen, (0, 255, 255), (int(car.x), int(car.y)), (int(end_x), int(end_y)), 2)

def render_car(car, screen):
    car_image_path = os.path.join('.', 'pygamecar', 'car.png')
    if not os.path.exists(car_image_path):
        raise FileNotFoundError(f"Car image file not found: {car_image_path}")
    car_image = pygame.image.load(car_image_path).convert_alpha()
    if car_image is None:
        raise ValueError(f"Failed to load car image from {car_image_path}")
    car_image = pygame.transform.scale(car_image, (20, 10))
    rotated_image = pygame.transform.rotate(car_image, -car.angle)
    rotated_rect = rotated_image.get_rect(center=(car.x, car.y))
    screen.blit(rotated_image, rotated_rect.topleft)
    render_rays(car, screen, 5, 300)  # Adjust num_rays and ray_length as needed

def render_game(screen, car):
    screen.fill((0, 0, 0))
    draw_track(screen)
    draw_gates(screen)
    render_car(car, screen)
    pygame.display.flip()
