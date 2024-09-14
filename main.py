import pygame
from car import create_car, controls, apply_friction, check_boundaries, check_lap_completion, draw_car
from track import draw_track, draw_start_finish_line
from utils import draw_lap_count, initialize_lap_counter
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, world
import os
import numpy as np
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

def run_sim(car, screen):
    lap_counter = initialize_lap_counter()
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
      
        controls(car, max_speed=100.0)
        apply_friction(car)
        check_boundaries(car)
        check_lap_completion(car, lap_counter)
        world.Step(1.0 / 60.0, 6, 2)
        
        screen.fill((0, 0, 0))
        draw_track(screen)
        draw_car(car, screen)
        draw_start_finish_line(screen)
        draw_lap_count(screen, lap_counter)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Box2D Car Racing")
    car = create_car()
    run_sim(car, screen)
