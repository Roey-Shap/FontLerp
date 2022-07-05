import pygame
import numpy as np


def mix_color(color1, color2, t):
    return (1-t)*np.array(color1) + t*np.array(color2)


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIME = pygame.Color("#55FF00")

GRAY = mix_color(BLACK, WHITE, 0.45)
LT_GRAY = mix_color(BLACK, WHITE, 0.75)

# class Color(object):
#     def Color(self, red, green, blue, alpha):
#         self.red = red
#         self.green = green
#         self.blue = blue
#         self.alpha = alpha
#         return
