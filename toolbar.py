import pygame
import custom_colors
import global_variables as globvar

class Toolbar(object):
    def __init__(self):
        w, h = globvar.SCREEN_DIMENSIONS
        self.width = w / 2
        self.height = h / 8

        margin = 6
        x_center = w/2
        y_center = 0 + margin
        self.left = x_center - self.width/2
        self.top = y_center
        self.border = 2
        return

    def draw(self, surface):
        alpha = 0.35
        s = pygame.Surface((self.width, self.height))  # the size of your rect
        s.set_alpha(round(alpha * 255))  # alpha level
        s.fill(custom_colors.LT_GRAY)
        surface.blit(s, (self.left, self.top))
        outline = pygame.Rect(self.left, self.top, self.width, self.height)
        pygame.draw.rect(surface, custom_colors.BLACK, outline, self.border, self.border, self.border, self.border)


