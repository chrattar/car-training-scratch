import pygame
import sys
from environment import Car, gates
from render import render_car, draw_track, draw_gates


pygame.init()
screen_width, screen_height = 800, 800 
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Racing Game")
clock = pygame.time.Clock()
car = Car()

#MAIN
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car.move(0)  # Accelerate
    if keys[pygame.K_DOWN]:
        car.move(1)  # Brake
    if keys[pygame.K_LEFT]:
        car.move(2)  # Turn left
    if keys[pygame.K_RIGHT]:
        car.move(3)  # Turn right

    if car.check_collision() or car.out_bounds():
        print("Collision detected. Resetting...")
        car.reset()


    screen.fill((0, 0, 0)) 
    draw_track(screen)
    draw_gates(screen)
    render_car(car, screen)
    pygame.display.flip()
    clock.tick(60)
