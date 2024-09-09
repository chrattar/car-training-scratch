import pygame

def draw_lap_count(screen, lap_counter):
    font = pygame.font.SysFont(None, 36)
    lap_text = font.render(f"Laps: {lap_counter['lap_count']}", True, (255, 255, 255))
    screen.blit(lap_text, (10, 10))

def initialize_lap_counter():
    return {'lap_count': 0, 'crossing': False}


# Start/Finish line definition
line_start_pos = (962, 100)
line_end_pos = (962, 198)
