import pygame
import custom_colors
import numpy as np
import cursor

# Constants
SCREEN_SIZE_FACTOR = 5
SCREEN_DIMENSIONS = np.array([160, 90]) * SCREEN_SIZE_FACTOR
FPS = 30
POINT_NP_DTYPE = np.float

empty_offset = np.array([0, 0], dtype=POINT_NP_DTYPE)

# Meta control variables
DEBUG = True

# Runtime control variables
pygame.init()
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)
clock = pygame.time.Clock()
cursor = cursor.Cursor()

origin = pygame.math.Vector2(0, 0)
selected_objects = []
abstract_points = []
curves = []
contours = []
glyphs = []

bezier_accuracy = 15
t_values = None

hovered_point = None
selected_point = None

# Input management
mouse_pos = None
mouse_click = None
mouse_held = False
mouse_click_left = False
mouse_scroll_directions = empty_offset

global_scale = 1
scroll_delta = 0.05
