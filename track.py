#track.py
import pygame
from config import screen, track_image, line_start_pos, line_end_pos

def draw_track(screen):
    screen.blit(track_image, (0, 0))

def draw_start_finish_line(screen):
    line_color = (45, 23, 44)  # Red color
    pygame.draw.line(screen, line_color, line_start_pos, line_end_pos, 5)
