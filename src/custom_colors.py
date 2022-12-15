import pygame
import numpy as np


def mix_color(color1, color2, t):
    color1 = to_numpy_color(color1)
    color2 = to_numpy_color(color2)
    return (1-t)*np.array(color1) + t*np.array(color2)


def to_numpy_color(color):
    if not isinstance(color, np.ndarray):
        return np.array(pygame.Color(color))
    return color


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIME = pygame.Color("#55FF00")
MUTED_BLUE = pygame.Color("#3B719F")
NEON_BLUE = pygame.Color("#04d9ff")
NEON_RED = pygame.Color("#ff1818")

GRAY = mix_color(BLACK, WHITE, 0.45)
LT_GRAY = mix_color(BLACK, WHITE, 0.75)
gray_value = 0.85 * 255
CONTOUR_FILL_GRAY = (gray_value, gray_value, gray_value)

# class Color(object):
#     def Color(self, red, green, blue, alpha):
#         self.red = red
#         self.green = green
#         self.blue = blue
#         self.alpha = alpha
#         return
