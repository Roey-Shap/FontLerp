import pygame
import global_variables as globvar
import custom_colors
import custom_math
import numpy as np

class Cursor(object):
    class Mode(int):
        NORMAL = 0
        PAN = 1

    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_prev = 0
        self.y_prev = 0

        self.panning = False
        self.panned_this_frame = False

        self.x_select_start = 0
        self.y_select_start = 0
        self.multiselecting = False

        self.mouse_pos = (0, 0)
        self.mouse_click = None
        self.mouse_held = False
        self.mouse_click_left = False

        return

    def update_mouse_variables(self):
        self.x_prev = self.x
        self.y_prev = self.y

        self.mouse_pos = globvar.mouse_pos
        self.mouse_click = globvar.mouse_click
        self.mouse_held = globvar.mouse_held
        self.mouse_click_left = globvar.mouse_click_left

        self.x = self.mouse_pos[0]
        self.y = self.mouse_pos[1]

        # keep pressing and mode separate - might want to toggle mode later
        self.panning = globvar.KEY_SPACE_PRESSED
        return

    def screen_to_world_space(self, screen_pos):
        screen_pos = np.array(screen_pos, dtype=globvar.POINT_NP_DTYPE)
        return (screen_pos / globvar.CAMERA_ZOOM) + globvar.CAMERA_OFFSET

    def world_to_screen_space(self, world_pos):
        world_pos = np.array(world_pos, dtype=globvar.POINT_NP_DTYPE)
        return (world_pos - globvar.CAMERA_OFFSET) * globvar.CAMERA_ZOOM

    def step(self, point_radius):
        # Update zoom and panning
        y_scroll = globvar.mouse_scroll_directions[1]
        if y_scroll != 0:
            prev_mouse_world_position = self.screen_to_world_space(self.mouse_pos)
            globvar.CAMERA_ZOOM *= 1 + (y_scroll * globvar.SCROLL_DELTA)
            globvar.CAMERA_ZOOM = custom_math.clamp(globvar.CAMERA_ZOOM, globvar.CAMERA_ZOOM_MIN, globvar.CAMERA_ZOOM_MAX)
            post_mouse_world_position = self.screen_to_world_space(self.mouse_pos)

            delta_mouse_pos = prev_mouse_world_position - post_mouse_world_position
            globvar.CAMERA_OFFSET += delta_mouse_pos

        self.step_normal(point_radius)

        if self.panning:
            self.step_pan()

        self.check_toolbar_hover()
        return

    def step_normal(self, point_radius):
        if not self.mouse_held:
            self.multiselecting = False

        if self.multiselecting:
            width = abs(self.x - self.x_select_start)
            height = abs(self.y - self.y_select_start)
            left = min(self.x, self.x_select_start)
            top = min(self.y, self.y_select_start)
            bounding_box = pygame.Rect(left, top, width, height)

        if not globvar.DEBUG:
            return
        ## From here on, we check mouse variables.
        ## If DEBUG is off, we don't do that
        hovered_point = globvar.hovered_point
        selected_point = globvar.selected_point

        # Allow the mouse to select objects
        # It should only access one at time to avoid issues of overlapping points
        if not self.mouse_held and hovered_point is not None:
            hovered_point.deselect()
            hovered_point = None

        # give priority to currently selected point to be hovered over
        if selected_point is not None:
            if selected_point.check_mouse_hover(point_radius, self.mouse_pos):
                hovered_point = selected_point

        # only bother checking if points can be dragged around if they're even shown
        if hovered_point is None and globvar.show_extra_curve_information:
            # only check points if you're available to select
            for point in globvar.abstract_points:
                if point.check_mouse_hover(point_radius, self.mouse_pos):
                    hovered_point = point
                    break

        if hovered_point is not None:
            # hovered_point.check_mouse_hover(point_radius, self.mouse_pos)
            hovered_point.handle_mouse_hover(self.mouse_pos, self.mouse_click_left, self.mouse_held)

        not_hovering_above_point = hovered_point is None
        if not_hovering_above_point and self.mouse_click_left:
            globvar.selected_point = None
            self.multiselecting = True
            self.x_select_start = self.x
            self.y_select_start = self.y

        globvar.hovered_point = hovered_point
        return

    def step_pan(self):
        # find the difference in the mouse's position and the current to determine
        # how far in world space we've moved

        delta_x = self.x - self.x_prev
        delta_y = self.y - self.y_prev

        if delta_x != 0 or delta_y != 0:
            self.panned_this_frame = True
            globvar.CAMERA_OFFSET[0] -= delta_x / globvar.CAMERA_ZOOM
            globvar.CAMERA_OFFSET[1] -= delta_y / globvar.CAMERA_ZOOM
        return

    def check_toolbar_hover(self):
        toolbar = globvar.toolbar
        for b in toolbar.buttons:
            b.check_mouse_hover(self.mouse_pos, self.mouse_click_left)

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