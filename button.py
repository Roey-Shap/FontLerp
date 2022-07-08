"""
Defines a Button object and its callback function.
"""

import pygame
import global_variables as globvar

import fonts
import custom_colors

class Button(object):
    def __init__(self, label, function, position, dimensions):
        self.label = label
        self.on_click = function
        self.x, self.y = position
        self.width, self.height = dimensions

        self.hovered = False
        self.bounding_box = pygame.Rect(self.x - self.width/2, self.y - self.height/2,
                                        self.x + self.width/2, self.y + self.height/2)
        self.held = False
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
        s = pygame.Surface((self.width, self.height))
        s.fill(self.get_color())
        surface.blit(s, (self.x, self.y))
        return


    def check_mouse_hover(self, mouse_pos):
        bounding_box = self.get_bounding_box()
        self.hovered = bounding_box.collidepoint(mouse_pos)
        return self.hovered

    def handle_mouse_hover(self, mouse_click_left):
        if self.hovered and mouse_click_left:
            self.on_click()
        return