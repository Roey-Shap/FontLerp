import bezier
import pygame
import custom_colors
import numpy as np
import cursor

# Constants
SCREEN_SIZE_FACTOR = 5
WIDTH = 160 * SCREEN_SIZE_FACTOR
HEIGHT = 90 * SCREEN_SIZE_FACTOR
FPS = 30

POINT_NP_DTYPE = np.float

# Meta control variables
DEBUG = False

# Runtime control variables
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
cursor = cursor.Cursor()

origin = pygame.math.Vector2(0, 0)
abstract_points = []
selected_objects = []
beziers = []

hovered_point = None

# Input management
mouse_pos = None
mouse_click = None
mouse_held = False
mouse_click_left = False
