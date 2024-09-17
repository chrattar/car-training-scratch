# config.py

# Screen and Rendering
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60  # Frames per second for rendering
RENDER_FREQUENCY = 1  # How often to render the screen (every frame, every Nth frame, etc.)

# Pygame Screen Initialization
def initialize_screen():
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Car Simulation")
    return screen

# Physics Parameters
GRAVITY = (0, 0)  # Gravity in Box2D world
TIME_STEP = 1.0 / FPS  # Box2D time step for the simulation
VELOCITY_ITERATIONS = 6  # Box2D velocity iterations for physics solver
POSITION_ITERATIONS = 2  # Box2D position iterations for physics solver

# Car Parameters
CAR_WIDTH = 3.0  # Car width in meters (Box2D units)
CAR_HEIGHT = 3.0  # Car height in meters (Box2D units)
CAR_INITIAL_POSITION = (96, 93)  # Initial position of the car in meters (Box2D units)
CAR_DENSITY = 1.0  # Car density for physics calculations
CAR_FRICTION = 0.3  # Car friction
CAR_COLOR = (255, 0, 0)  # Car color for rendering

# Agent Parameters
MAX_SPEED = 50  # Maximum speed the car can reach in meters per second
LEARNING_RATE = 0.001  # Learning rate for the agent's model
DISCOUNT_FACTOR = 0.99  # Discount factor for future rewards
EPSILON_START = 1.0  # Initial epsilon for epsilon-greedy policy
EPSILON_MIN = 0.01  # Minimum epsilon for epsilon-greedy policy
EPSILON_DECAY = 0.995  # Decay rate for epsilon after each episode

# Environment Parameters
NUM_EPISODES = 1000  # Number of episodes to train the agent
MAX_STEPS_PER_EPISODE = 500  # Maximum steps per episode before reset
TRACK_BOUNDARIES = [  # Example track boundary points in meters (Box2D units)
    [(0, 0), (192, 0), (192, 108), (0, 108)]
]

# Vision Parameters
VISION_MAX_DISTANCE = 1000  # Maximum distance for the car's vision in meters
VISION_NUM_SAMPLES = 40  # Number of samples for vision ray casting

# Helper Imports (to avoid circular dependencies in some IDEs)
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
screen = initialize_screen()  # Initialize screen for rendering

# Rendering Frequency for Training Visualization
def should_render(frame_count):
    # Function to decide if the current frame should be rendered based on frequency
    return frame_count % RENDER_FREQUENCY == 0
