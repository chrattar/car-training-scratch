#car-data.py
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody, b2Vec2
import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((1920, 1080))

def create_car(world):
    car_body_def = b2BodyDef(type=b2_dynamicBody, position=(96, 93), allowSleep=True)
    car = world.CreateBody(car_body_def)
    car_shape = b2PolygonShape(box=(3.0, 1.5))
    car.CreateFixture(shape=car_shape, density=1.0, friction=0.3)
    return car

def controls(car, max_speed):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car.ApplyForceToCenter(car.GetWorldVector((1, 0)) * 10000, True)
    if keys[pygame.K_DOWN]:
        car.ApplyForceToCenter(car.GetWorldVector((-1, 0)) * 5000, True)
    if keys[pygame.K_LEFT]:
        car.ApplyTorque(10.0, True)
    elif keys[pygame.K_RIGHT]:
        car.ApplyTorque(-10.0, True)
    else:
        car.angularVelocity *= 0.8
    if car.linearVelocity.length > max_speed:
        car.linearVelocity *= max_speed / car.linearVelocity.length

