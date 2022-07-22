"""
Defines a Button object and its callback function.
"""

import pygame
import global_variables as globvar

import fonts
import custom_colors
import custom_pygame
import ptext

class Button(object):
    COLOR_IDLE = custom_colors.LT_GRAY
    COLOR_CLICK = custom_colors.GRAY
    COLOR_HOVER = custom_colors.mix_color(COLOR_IDLE, COLOR_CLICK, 0.5)
    def __init__(self, label, information, function, position, dimensions):
        self.label = label
        self.information = information
        self.on_click = function
        self.x, self.y = position
        self.position = (self.x, self.y)
        self.width, self.height = dimensions

        self.hovered = False
        self.bounding_box = pygame.Rect(self.x, self.y,
                                        self.width, self.height)
        self.held = False
        self.clicked = False

        self.border_width = 1
        self.border_rad = 3
        return

    def get_bounding_box(self):
        return self.bounding_box

    def get_color(self):
        # in decreasing priority; if you're held, you're also hovered, but should show held
        if self.held or self.clicked:
            return self.COLOR_CLICK
        if self.hovered:
            return self.COLOR_HOVER
        return self.COLOR_IDLE


    def draw(self, surface):
        outline = pygame.Rect(self.x, self.y, self.width, self.height)

        alpha = 0.5
        col = self.get_color()
        col = (col[0], col[1], col[2], alpha * 255)
        custom_pygame.draw_rect_alpha(surface, col, outline)

        pygame.draw.rect(surface, custom_colors.BLACK, outline, self.border_width, self.border_rad, self.border_rad,
                         self.border_rad, self.border_rad)

        centered_pos = (self.position[0] + self.width/2, self.position[1] + self.height/2)
        tsurf, tpos = ptext.draw(self.label, color=custom_colors.BLACK, width=self.width, lineheight=1.5, center=centered_pos)
        surface.blit(tsurf, tpos)
        return


    def check_mouse_hover(self, mouse_pos, mouse_click_left):
        bounding_box = self.get_bounding_box()
        self.hovered = bounding_box.collidepoint(mouse_pos)
        if self.hovered and mouse_click_left:
            self.on_click()
        return self.hovered
