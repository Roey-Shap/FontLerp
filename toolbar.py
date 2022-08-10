import pygame
import custom_colors
import global_variables as globvar
import global_manager
import button
import numpy as np


class Toolbar(object):
    def __init__(self):
        w, h = globvar.SCREEN_DIMENSIONS
        self.width = w / 2
        self.height = h / 8

        margin = 3
        self.button_horizontal_buffer = 3
        x_center = w/2
        y_center = 0 + margin
        self.left = x_center - self.width/2
        self.top = y_center
        self.border_rad = 5
        self.border_width = 1
        self.top_left = np.array([self.left, self.top])
        self.standard_button_dimensions = np.array([self.width/8, self.height])
        self.where_last_button_ends = 0

        self.buttons = []
        toggle_debug_info_button = self.add_button("Toggle Debug Info Display",
                                                    "Click to toggle debug information",
                                                    global_manager.toggle_debug_info,
                                                    self.standard_button_dimensions)
        toggle_mapping_display_button = self.add_button("Toggle Mapping Display",
                                                        "Click to view the mapping between the glyphs, if one exists",
                                                        global_manager.toggle_show_current_glyph_mapping,
                                                        self.standard_button_dimensions)
        toggle_control_point_display_button = self.add_button("Toggle Control Point Display",
                                                              "Click to view extra curve information",
                                                              global_manager.toggle_show_extra_curve_information,
                                                              self.standard_button_dimensions)
        toggle_test_mixed_glyph_button = self.add_button("Toggle Morphing Glyph",
                                                     "Shows a glyph displaying all interpolation states of a test letter",
                                                     global_manager.toggle_lerped_glyph_display,
                                                     self.standard_button_dimensions)

        make_reduction_mapping_button = self.add_button("Suggest Reduction Mapping",
                                                        "Make a mapping between the two active glyphs",
                                                        global_manager.make_mapping_from_active_glyphs,
                                                        self.standard_button_dimensions)
        point_manipulation_button = self.add_button("Manipulate Points",
                                                     "Drag points around",
                                                     global_manager.go_into_point_manipulation_mode,
                                                     self.standard_button_dimensions)

        make_custom_mapping_button = self.add_button("Create New Custom Reduction Mapping",
                                                     "Make your own mapping between the two active glyphs",
                                                     global_manager.add_custom_mapping,
                                                     self.standard_button_dimensions)

        return

    def add_button(self, title, info, function, dimensions):
        upper_left_corner = np.array([self.where_last_button_ends + self.button_horizontal_buffer, 0]) + self.top_left
        b = button.Button(title, info, function, upper_left_corner, dimensions)
        self.where_last_button_ends += dimensions[0] + self.button_horizontal_buffer

        self.buttons.append(b)
        return b

    def draw(self, surface):
        alpha = 0.35
        s = pygame.Surface((self.width, self.height))  # the size of your rect
        s.set_alpha(round(alpha * 255))  # alpha level
        s.fill(custom_colors.LT_GRAY)
        surface.blit(s, (self.left, self.top))
        outline = pygame.Rect(self.left, self.top, self.width, self.height)
        pygame.draw.rect(surface, custom_colors.BLACK, outline, self.border_width, self.border_rad, self.border_rad, self.border_rad, self.border_rad)

        for b in self.buttons:
            b.draw(surface)

        return

