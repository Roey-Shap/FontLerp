import pygame
import custom_colors
import numpy as np
import cursor
import glyph

# Functions
def em_to_worldspace(np_array):
    return np_array / (GLYPH_UNIT_TO_WORLD_SPACE_RATIO*CAMERA_ZOOM)


# Constants
POINT_NP_DTYPE = np.float
empty_offset = np.array([0, 0], dtype=POINT_NP_DTYPE)

SCREEN_SIZE_FACTOR = 7
SCREEN_DIMENSIONS = np.array([160, 90]) * SCREEN_SIZE_FACTOR
FPS = 30

CAMERA_ZOOM = 1
CAMERA_OFFSET = empty_offset.copy()
CAMERA_ZOOM_MIN = 0.0001
CAMERA_ZOOM_MAX = 20

GLYPH_UNIT_TO_WORLD_SPACE_RATIO = 4

CONTOUR_CURVE_AMOUNT_THRESHOLD = 30
DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS = em_to_worldspace(np.array([1200, 1300], dtype=POINT_NP_DTYPE))
DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT = SCREEN_DIMENSIONS * np.array([1/5, 1/3])

# Meta constants
DEBUG_MESSAGE_POSITION = SCREEN_DIMENSIONS * np.array([0.5, 0.2])
SCROLL_DELTA = 0.025
POINT_DRAW_RADIUS = 3
LINE_THICKNESS = 3
POINTS_TO_CHECK_AVERAGE_WITH = 50                   # for the whole glyph we're getting the average for!
POINTS_TO_GET_CONTOUR_MAPPING_WITH = 150            # for EACH contour
EM_TO_FONT_SCALE = 0.07

up_lefters = "cegikmnoswx"
up_righters = "fr"
down_righters = "bhjlpqyz"
down_lefters = "adtuv"


# Shape constants
CIRCLE_CONST = 0.552284749831



# Meta control variables
DEBUG = False
update_screen = True

hovered_point = None
selected_point = None

mouse_pos = [0, 0]
mouse_click = None
mouse_held = False
mouse_click_left = False
mouse_scroll_directions = empty_offset

KEY_SPACE_HELD = False
KEY_SPACE_PRESSED = False

show_current_glyph_mapping = False
show_extra_curve_information = False
show_mixed_glyph = True
show_lerped_glyph_text = True

BEZIER_ACCURACY = 4
t_values = None

active_glyphs = [None, None]
current_glyph_mapping = None
current_glyph_mapping_is_valid = False
current_glyph_mapping_method = None
current_glyph_lerping_method = None
lerped_glyph = None

debug_width_points = []         # for when drawing a piece of text and finding character separators

# Runtime control variables
pygame.init()
screen = None
clock = None
cursor = None
toolbar = None
manager = None

origin = pygame.math.Vector2(0, 0)
selected_objects = []
abstract_points = []
curves = []
contours = []
glyphs = []
glyph_mappings = {}            # a dictionary of key:value pairs (g1, g2): [custom_map, reduction_map, insertion_map]


test_curves = []
marking_test_points_lists = []

random_debug_points = []

cur_character_lerped = "X"