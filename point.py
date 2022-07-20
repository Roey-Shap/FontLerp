import numpy as np
import pygame
import custom_colors
import global_variables as globvar
import fonts
import custom_math

"""
An abstract representation of a Point to be manipulated by the mouse; coordinates are in WORLDSPACE ("true position")
"""
class Point(object):
    COLOR_IDLE = custom_colors.BLACK
    COLOR_HOVER = custom_colors.GRAY
    COLOR_CLICK = custom_colors.LT_GRAY

    def __init__(self, coords, is_endpoint=False):
        self.x = coords[0]
        self.y = coords[1]

        self.hovered = False
        self.clicked = False
        self.held = False

        self.changed_position_this_frame = False
        self.prev_coords = self.np_coords()

        self.is_endpoint = is_endpoint
        self.base_color = custom_colors.BLACK if is_endpoint else custom_colors.WHITE
        return

    def destroy(self):
        index = globvar.abstract_points.index(self)
        globvar.abstract_points.pop(index)
        return

    def check_mouse_hover(self, r, mouse_pos):
        self.changed_position_this_frame = False
        self.prev_coords = self.np_coords()

        bounding_box = custom_math.get_bounding_box_from_radius(self.x, self.y, r)
        self.hovered = bounding_box.collidepoint(mouse_pos)
        return self.hovered

    def handle_mouse_hover(self, mouse_pos, mouse_click_left, mouse_held):
        if self.held:
            self.update_coords(mouse_pos)
            globvar.selected_point = self
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
    def draw(self, surface, radius, debug_info=False):
        if not self.is_endpoint:
            radius *= 0.75
        color = custom_colors.mix_color(self.base_color, self.get_color(), 0.5)
        pygame.draw.circle(surface, color, self.np_coords(), radius)

        if debug_info:
            text_rect = pygame.Rect(0, 0, 100, 100)
            debug_text_surface = fonts.multiLineSurface("x: " + str(round(self.x)) + ", y: " + str(round(self.y)), fonts.FONT_DEBUG, text_rect,
                                                        custom_colors.BLACK)
            surface.blit(debug_text_surface, (self.np_coords()))

        if globvar.selected_point == self:
            debug_alpha = 0.45
            s = pygame.Surface((radius*2, radius*2))  # the size of your rect
            s.set_alpha(np.floor(debug_alpha * 255))  # alpha level
            s.fill(custom_colors.LIME)
            surface.blit(s, (self.x - radius, self.y - radius))

    def get_color(self):
        # in decreasing priority; if you're held, you're also hovered, but should show held
        if self.held or self.clicked:
            return self.COLOR_CLICK
        if self.hovered:
            return self.COLOR_HOVER
        return self.COLOR_IDLE

