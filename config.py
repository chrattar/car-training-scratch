#config.py
import pygame
from Box2D import b2World
import os
from Box2D import b2World, b2BodyDef, b2PolygonShape, b2_dynamicBody, b2Vec2
import pygame
from utils import line_end_pos, line_start_pos
import numpy as np

# Screen dimensions
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# World Variables
world = b2World(gravity=(0, 0))

# Assets
current_dir = os.path.dirname(__file__)  # Get the current directory of the script
car_image_path = os.path.join(current_dir, 'car.png')  # Construct full path to 'car.png'
track_image_path = os.path.join(current_dir, 'track.png')  # Construct full path to 'track.png'

# Load and scale assets
car_image = pygame.image.load(car_image_path)
car_image = pygame.transform.scale(car_image, (30, 36))
track_image = pygame.image.load(track_image_path)

# Start/Finish line definition
line_start_pos = (962, 100)
line_end_pos = (962, 259)
