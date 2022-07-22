import numpy as np
import pygame
import global_variables as globvar

import operator as op
from functools import reduce

"""
Maps a value in [pre_min, pre_max] to another in [post_min, post_max]
from https://www.reddit.com/r/gamemaker/comments/6nk3us/help_is_there_a_gamemaker_equivalent_to_the_map/
"""
def map(value, pre_min, pre_max, post_min, post_max):
    prop = (value - pre_min) / (pre_max - pre_min);
    return (prop * (post_max - post_min)) + post_min;


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom



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


def weighted(nb):
    if nb is None:
        return float('inf')
    else:
        return nb


def sort_dictionary(dictionary, by_key_or_value=0, reverse=False):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: weighted(item[by_key_or_value]), reverse=reverse)}

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


"""
Returns +1 if the points move in clockwise fashion and -1 if they move in counter-clockwise fashion
"""
def points_clock_direction(points):
    num_points = len(points)
    signed_area = 0
    x1, y1 = points[0]
    x2 = x1
    y2 = y1
    for i in range(num_points):
        x1 = x2
        y1 = y2
        x2, y2 = points[i]

        signed_area += (x1 * y2) - (x2 * y1)

    # don't forget about the cyclic last point!
    x1 = x2
    y1 = y2
    x2, y2 = points[0]
    signed_area += (x1 * y2) - (x2 * y1)

    return np.sign(signed_area)
