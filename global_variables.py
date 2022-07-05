import bezier
import pygame
import custom_colors
import numpy as np

SCREEN_SIZE_FACTOR = 5
WIDTH = 160 * SCREEN_SIZE_FACTOR
HEIGHT = 90 * SCREEN_SIZE_FACTOR
FPS = 30

POINT_NP_DTYPE = np.float

DEBUG = False

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

mouse_pos = None
mouse_click = None
mouse_held = False
mouse_click_left = False


origin = pygame.math.Vector2(0, 0)
