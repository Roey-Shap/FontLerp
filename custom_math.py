import pygame
import global_variables as globvar


"""
Maps a value in [pre_min, pre_max] to another in [post_min, post_max]
from https://www.reddit.com/r/gamemaker/comments/6nk3us/help_is_there_a_gamemaker_equivalent_to_the_map/
"""
def map(value, pre_min, pre_max, post_min, post_max):
    prop = (value - pre_min) / (pre_max - pre_min);
    return (prop * (post_max - post_min)) + post_min;


def interpolate_np(ndarr1, ndarr2, t):
    weighted_1 = ndarr1 * t
    weighted_2 = ndarr2 * (1-t)
    return weighted_1 + weighted_2


def get_bounding_box_from_radius(x, y, r):
    return pygame.Rect(x - r, y - r, r * 2, r * 2)


def camera_to_worldspace(np_array):
    return (np_array / globvar.CAMERA_ZOOM) + globvar.CAMERA_OFFSET


def world_to_cameraspace(np_array):
    return (np_array - globvar.CAMERA_OFFSET) * globvar.CAMERA_ZOOM

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))
