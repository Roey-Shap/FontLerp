import global_variables as globvar
import numpy as np
import pygame
import custom_colors
import ptext
import glyph

class GlobalManager(object):
    class State(int):
        NO_TOOL = 0
        ADJUSTING_POINTS = 1
        ADDING_CURVES = 2
        MAPPING_CURVES = 3

    class MappingCurveState(int):
        OMIN = 0
        INSERTION = 1


    def __init__(self):
        self.state = self.State.ADJUSTING_POINTS

        return

    def step(self):
        # synchronize Cursor object with inputs
        globvar.cursor.update_mouse_variables()
        globvar.cursor.step(globvar.POINT_DRAW_RADIUS)

        if self.state == self.State.ADJUSTING_POINTS:
            self.step_adjusting_points()
        if self.state == self.State.MAPPING_CURVES:
            self.step_mapping_curves()


    def step_adjusting_points(self):
        return


    def step_mapping_curves(self):
        return


    """
    Draw lines between each of the beziers based on the mappings provided
    """
    def draw_mapping_reduction(self, surface):
        if globvar.current_glyph_mapping is None:
            pos = (globvar.DEBUG_MESSAGE_POSITION[0], globvar.DEBUG_MESSAGE_POSITION[1])
            tsurf, tpos = ptext.draw("No mapping currently active...",
                                     pos,
                                     color=custom_colors.BLACK)
            surface.blit(tsurf, tpos)
            return

        glyph1, glyph2 = globvar.active_glyphs
        mappings = globvar.current_glyph_mapping

        n1 = len(glyph1)
        n2 = len(glyph2)

        if n2 > n1:
            raise AttributeError("Glyph 1 needs to have at least as many curves as Glyph 2 to draw the correct mapping")

        start_color = custom_colors.NEON_BLUE
        end_color = custom_colors.NEON_RED
        total_mappings = len(mappings)
        # iterate through each contour of G1 and show the curve that it's mapped to in G2
        for g1_contour_index in mappings:
            g2_contour_index, c1_to_c2_mapping = mappings[g1_contour_index]
            current_mapped_contours_color = custom_colors.mix_color(start_color, end_color, g1_contour_index / total_mappings)
            if g1_contour_index == total_mappings - 1:
                current_mapped_contours_color = end_color

            g1_contour = glyph1.contours[g1_contour_index]
            g2_contour = glyph2.contours[g2_contour_index]

            # take each pair in the mapping; if it's not a null, draw a line
            for pair in c1_to_c2_mapping.items():
                c1_index, c2_index = pair
                if c2_index is not None:
                    c1_curve_center = g1_contour.curves[c1_index].get_center_camera()
                    c2_curve_center = g2_contour.curves[c2_index].get_center_camera()
                    pygame.draw.line(surface, current_mapped_contours_color, c1_curve_center, c2_curve_center)

                # draw over the current contour outline with this color
                g1_contour.draw(surface, globvar.POINT_DRAW_RADIUS, [current_mapped_contours_color]*3)
                g2_contour.draw(surface, globvar.POINT_DRAW_RADIUS, [current_mapped_contours_color]*3)


def set_active_glyphs(g1, g2):
    # check if the tuple (g1, g2) is not in the dictionary already; in that case, we'll initialize them
    pair_has_been_active_before = False
    for key in globvar.glyph_mappings:
        if key[0] == g1 or key[1] == g2:
            pair_has_been_active_before = True
            break

    if not pair_has_been_active_before:
        globvar.glyph_mappings[(g1, g2)] = [None, None, None]

    globvar.active_glyphs = [g1, g2]
    return

def toggle_show_current_glyph_mapping():
    globvar.show_current_glyph_mapping = not globvar.show_current_glyph_mapping
    return

def toggle_show_extra_curve_information():
    globvar.show_extra_curve_information = not globvar.show_extra_curve_information
    return

def make_mapping_from_active_glyphs():
    g1, g2 = globvar.active_glyphs
    globvar.current_glyph_mapping, globvar.glyph_score = glyph.find_glyph_null_contour_mapping(g1, g2, debug_info=True)
    insert_reduction_glyph_mapping(g1, g2, globvar.current_glyph_mapping)
    globvar.current_glyph_mapping_is_valid = True
    return

def go_into_point_manipulation_mode():
    globvar.manager.state = GlobalManager.State.ADJUSTING_POINTS
    # globvar.show_extra_curve_information = True
    return

def add_custom_mapping():
    # put the manager into mapping creation mode
    globvar.manager.state = GlobalManager.State.MAPPING_CURVES
    # globvar.show_extra_curve_information = False

    # store new custom mapping, set up other global variables surrounding validity of glyph being drawn, etc.
    current_g1, current_g2 = globvar.active_glyphs

    # a mapping consists of a dictionary of (contour in g1, (contour in g2, mapping details from c1 -> c2))
    new_mapping = {c1_index: (None, None) for c1_index in range(len(current_g1))}

    # add this glyph mapping to the list of mappings attributed to these two glyphs
    insert_custom_glyph_mapping(current_g1, current_g2, new_mapping)
    globvar.current_glyph_mapping = new_mapping
    globvar.current_glyph_mapping_is_valid = False
    globvar.active_glyphs = [None, None]
    return


def insert_custom_glyph_mapping(g1, g2, mapping):
    globvar.glyph_mappings[(g1, g2)][0] = mapping
    return


def insert_reduction_glyph_mapping(g1, g2, mapping):
    globvar.glyph_mappings[(g1, g2)][1] = mapping
    return


def insert_insertion_glyph_mapping(g1, g2, mapping):
    globvar.glyph_mappings[(g1, g2)][2] = mapping
    return


    """
    Based on the level of zoom, adjust the accuracy with which Bezier curves are drawn.
    """
def update_bezier_accuracy():
    globvar.BEZIER_ACCURACY = int(1.8 * np.log2(globvar.CAMERA_ZOOM * 10)) + 2
    return globvar.BEZIER_ACCURACY

def calculate_t_array():
    globvar.t_values = np.array([[(i / globvar.BEZIER_ACCURACY) ** 3,
                                  (i / globvar.BEZIER_ACCURACY) ** 2,
                                  (i / globvar.BEZIER_ACCURACY), 1] for i in range(globvar.BEZIER_ACCURACY + 1)],
                                dtype=globvar.POINT_NP_DTYPE)
    return globvar.t_values
