import pygame
import global_variables as globvar
import custom_colors

class Cursor(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_prev = 0
        self.y_prev = 0
        self.x_select_start = 0
        self.y_select_start = 0
        self.multiselecting = False

        self.mouse_pos = (0, 0)
        self.mouse_click = None
        self.mouse_held = False
        self.mouse_click_left = False
        return

    def update_mouse_variables(self):
        self.mouse_pos = globvar.mouse_pos
        self.mouse_click = globvar.mouse_click
        self.mouse_held = globvar.mouse_held
        self.mouse_click_left = globvar.mouse_click_left

        self.x = self.mouse_pos[0]
        self.y = self.mouse_pos[1]

    def step(self, point_radius):
        if not self.mouse_held:
            self.multiselecting = False

        if self.multiselecting:
            width = abs(self.x - self.x_select_start)
            height = abs(self.y - self.y_select_start)
            left = min(self.x, self.x_select_start)
            top = min(self.y, self.y_select_start)
            bounding_box = pygame.Rect(left, top, width, height)


        hovered_point = globvar.hovered_point

        # Allow the mouse to select objects
        # It should only access one at time to avoid issues of overlapping points
        if not self.mouse_held and hovered_point is not None:
            hovered_point.deselect()
            hovered_point = None

        if hovered_point is None:
            # only check points if you're available to select
            for point in globvar.abstract_points:
                if point.check_mouse_hover(point_radius, self.mouse_pos):
                    hovered_point = point
                    break

        if hovered_point is not None:
            hovered_point.handle_mouse_hover(self.mouse_pos, self.mouse_click_left, self.mouse_held)

        can_multiselect = hovered_point is None
        if can_multiselect and self.mouse_click_left:
            self.multiselecting = True
            self.x_select_start = self.x
            self.y_select_start = self.y

        globvar.hovered_point = hovered_point
        return

    def draw(self, surface):
        if self.multiselecting:
            width = abs(self.x - self.x_select_start)
            height = abs(self.y - self.y_select_start)
            left = min(self.x, self.x_select_start)
            top = min(self.y, self.y_select_start)

            alpha = 0.2
            s = pygame.Surface((width, height))  # the size of your rect
            s.set_alpha(round(alpha * 255))  # alpha level
            s.fill(custom_colors.LT_GRAY)
            surface.blit(s, (left, top))

        return