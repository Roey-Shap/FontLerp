import numpy as np
import pygame
import custom_colors
import global_variables as globvar

# An abstract representation of a Point to be manipulated by the mouse
import global_variables


class Point(object):
    COLOR_IDLE = custom_colors.BLACK
    COLOR_HOVER = custom_colors.GRAY
    COLOR_CLICK = custom_colors.LT_GRAY

    def __init__(self, coords):
        self.x = coords[0]
        self.y = coords[1]

        self.hovered = False
        self.clicked = False
        self.held = False

        self.changed_position_this_frame = False
        self.prev_coords = self.np_coords()
        return

    def __len__(self):
        return 2

    def check_mouse_hover(self, r, mouse_pos):
        bounding_box = pygame.Rect(self.x - r, self.y - r, r * 2, r * 2)
        self.hovered = bounding_box.collidepoint(mouse_pos)
        return self.hovered

    def handle_mouse_hover(self, mouse_pos, mouse_click_left, mouse_held):
        changed_position_this_frame = False
        if self.held:
            self.prev_coords = self.np_coords()
            self.update_coords(mouse_pos)
        self.clicked = self.hovered and mouse_click_left
        self.held = (self.held or self.clicked) and mouse_held

        if self.held:
            self.changed_position_this_frame = not np.all(np.isclose(self.prev_coords, self.np_coords()))
        return

    def deselect(self):
        self.held = False
        return

    def update_coords(self, coords):
        self.x = coords[0]
        self.y = coords[1]
        return

    def np_coords(self):
        return np.array([self.x, self.y], dtype=globvar.POINT_NP_DTYPE)

# Drawing Functions
    def get_color(self):
        # in decreasing priority; if you're held, you're also hovered, but should show held
        if self.held or self.clicked:
            return self.COLOR_CLICK
        if self.hovered:
            return self.COLOR_HOVER
        return self.COLOR_IDLE

    def draw(self, surface, radius, base_color=custom_colors.BLACK):
        color = custom_colors.mix_color(base_color, self.get_color(), 0.75)
        pygame.draw.circle(surface, color, (self.x, self.y), radius)

        if globvar.DEBUG:
            debug_alpha = 0.4
            s = pygame.Surface((radius*2, radius*2))  # the size of your rect
            s.set_alpha(np.floor(debug_alpha * 255))  # alpha level
            s.fill(custom_colors.LIME)
            surface.blit(s, (self.x - radius, self.y - radius))

    #
    # def __add__(self, other):
    #     return self.coords + other.coords
    #
    # def __sub__(self, other):
    #     return self.coords - other.coords
    #
    # def __mul__(self, other):
    #     return self.coords - other
    #

