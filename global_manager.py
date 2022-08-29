import numpy as np
import pygame

import contour
import glyph

import custom_math
import global_variables as globvar
import custom_colors
import ptext

import ttfConverter
import toolbar
import cursor

"""
Represents the global manager object, which controls which state the editor is in.
For example, is the user manipulating points? If so, how does that change the effect of their mouse input?

Also manages assigning glyphs as active and interpolating between them
"""
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

        update_bezier_accuracy()
        calculate_t_array()


        # instantiate meta objects
        globvar.manager = self

        globvar.screen = pygame.display.set_mode(globvar.SCREEN_DIMENSIONS)
        globvar.clock = pygame.time.Clock()
        globvar.cursor = cursor.Cursor()

        globvar.toolbar = toolbar.Toolbar()
        pygame.init()
        pygame.display.set_caption("Font Interpolater")

        return

    def step(self):
        # synchronize Cursor object with inputs
        globvar.cursor.update_mouse_variables()
        globvar.cursor.step(globvar.POINT_DRAW_RADIUS)

        if self.state == self.State.ADJUSTING_POINTS:
            self.step_adjusting_points()
        if self.state == self.State.MAPPING_CURVES:
            self.step_mapping_curves()

    # @TODO
    def step_adjusting_points(self):
        return

    # @TODO
    def step_mapping_curves(self):
        return


# @TODO Implemented a while ago - needs to be redone
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

"""
Takes in a string and sets the mapping and interpolation methods accordingly.
"""
def set_mapping_and_lerping_methods(string):
    if string == "Pillow Projection":
        globvar.current_glyph_mapping_method = contour.find_pillow_projection_mapping
        globvar.current_glyph_lerping_method = contour.lerp_contours_pillow_proj
    elif string == "Relative Projection":
        globvar.current_glyph_mapping_method = contour.find_relative_projection_mapping
        globvar.current_glyph_lerping_method = contour.lerp_contours_relative_proj
    else:       # default case is Pillow Projection
        raise ValueError("No default mapping and lerping method!")
        # print("Warning! Using a DEFAULT ")
        # globvar.current_glyph_mapping_method = contour.find_pillow_projection_mapping
        # globvar.current_glyph_lerping_method = contour.lerp_contours_pillow_proj

    return

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


# Button functions ###

def toggle_show_current_glyph_mapping():
    globvar.show_current_glyph_mapping = not globvar.show_current_glyph_mapping
    return

def toggle_show_extra_curve_information():
    globvar.show_extra_curve_information = not globvar.show_extra_curve_information
    return

def toggle_lerped_glyph_display():
    globvar.show_mixed_glyph = not globvar.show_mixed_glyph
    return

def toggle_debug_info():
    globvar.DEBUG = not globvar.DEBUG
    return

def split_test_debug_curve():
    split_curves = []
    for g in globvar.test_curves:
        for cont in g.contours:
            for c in cont.curves:
                piece1, piece2 = c.de_casteljau(0.5)
                split_curves.append(glyph.glyph_from_curves([piece1]))
                split_curves.append(glyph.glyph_from_curves([piece2]))

    globvar.test_curves = split_curves

def toggle_lerped_text_display():
    globvar.show_lerped_glyph_text = not globvar.show_lerped_glyph_text
    return

# #######

def make_mapping_from_active_glyphs():
    g1, g2 = globvar.active_glyphs
    globvar.current_glyph_mapping, globvar.glyph_score = glyph.find_glyph_contour_mapping(g1, g2, globvar.current_glyph_mapping_method)
    # insert_reduction_glyph_mapping(g1, g2, globvar.current_glyph_mapping)
    globvar.current_glyph_mapping_is_valid = True
    return


def lerp_active_glyphs(t):
    g1, g2 = globvar.active_glyphs
    globvar.lerped_glyph = glyph.lerp_glyphs(g1, g2,
                                             globvar.current_glyph_lerping_method,
                                             globvar.current_glyph_mapping, t)
    globvar.lerped_glyph.worldspace_offset_by(-globvar.lerped_glyph.get_upper_left_world())
    globvar.lerped_glyph.update_all_parameters()

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
    # globvar.active_glyphs = [None, None]
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

def insert_pillow_projection_glyph_mapping(g1, g2, mapping):
    globvar.glyph_mappings[(g1, g2)][3] = mapping


    """
    Based on the level of zoom, adjust the accuracy with which Bezier curves are drawn.
    """
def update_bezier_accuracy():
    globvar.BEZIER_ACCURACY = int(1.4 * np.log2(globvar.CAMERA_ZOOM * 10)) + 2
    return globvar.BEZIER_ACCURACY

def calculate_t_array():
    globvar.t_values = np.array([[(i / globvar.BEZIER_ACCURACY) ** 3,
                                  (i / globvar.BEZIER_ACCURACY) ** 2,
                                  (i / globvar.BEZIER_ACCURACY), 1] for i in range(globvar.BEZIER_ACCURACY + 1)],
                                dtype=globvar.POINT_NP_DTYPE)
    return globvar.t_values


def get_glyphs_from_text(text, font1, font2, wrap_x=None):
    lines = text.splitlines()

    # first find how many mappings we need; one for each character present in the text
    characters_in_text = custom_math.unique_string_values(text)
    character_mappings = {}

    widest_glyph_width = 0
    tallest_glyph_height = 0
    for char in characters_in_text:
        print("Finding mapping for:", char)
        print("Doing Font:", font1)
        g1 = ttfConverter.glyph_from_font(char, font1)
        print("Doing Font:", font2)
        g2 = ttfConverter.glyph_from_font(char, font2)
        print("")

        widest_glyph_width = max(widest_glyph_width, g1.width, g2.width)
        tallest_glyph_height = max(tallest_glyph_height, g1.height, g2.height)

        char_mapping, score = glyph.find_glyph_contour_mapping(g1, g2, globvar.current_glyph_mapping_method)
        character_mappings[char] = (g1, g2, char_mapping)


    # Begin building characters

    string_length = sum(len(line) for line in lines)  # note that this includes spaces, so the lerping has
    # higher visual correlation with distance along the line

    between_character_buffer = 1
    current_character_count = 0
    current_draw_x = 0
    current_draw_y = 0
    starting_line = True
    lerped_glyphs = []

    # now we have all of the mappings we need - let's generate the text one character at a time
    globvar.debug_width_points = []
    for line in lines:
        for char in line:
            globvar.debug_width_points.append([current_draw_x, current_draw_y])
            t = current_character_count / string_length
            if char == ' ':
                if not starting_line:
                    current_draw_x += widest_glyph_width * globvar.EM_TO_FONT_SCALE
            else:
                starting_line = False

                # get mapping info
                g1, g2, char_mapping = character_mappings[char]
                globvar.cur_character_lerped = char                     # TODO DEBUG INFORMATION, CAN GET RID OF

                # generate the correct tween-glyph and move the draw_x accordingly
                lerped_glyph = glyph.lerp_glyphs(g1, g2, globvar.current_glyph_lerping_method, char_mapping, t)
                lerped_glyph.worldspace_scale_by(globvar.EM_TO_FONT_SCALE)
                lerped_glyph.update_bounds()

                # make its left side aligned with (0, 0) and then push as needed
                offset = np.array([current_draw_x, current_draw_y]) - lerped_glyph.get_upper_left_world()
                lerped_glyph.worldspace_offset_by(offset)
                lerped_glyph.update_bounds()

                lerped_glyphs.append(lerped_glyph)
                current_draw_x += lerped_glyph.width + between_character_buffer

            if wrap_x is not None and current_draw_x >= wrap_x:
                current_draw_x = 0
                current_draw_y += tallest_glyph_height * globvar.EM_TO_FONT_SCALE
                starting_line = True

            current_character_count += 1

        # end line loop by resetting current horizontal position
        current_draw_x = 0
        current_draw_y += tallest_glyph_height * globvar.EM_TO_FONT_SCALE
        starting_line = True


    return lerped_glyphs

def draw_lerped_text(surface, lerped_glyphs):
    for g in lerped_glyphs:
        g.draw(surface, globvar.POINT_DRAW_RADIUS, color=(0, 0, 0))
    return