import numpy as np
import pygame
import math

import global_variables as globvar
import custom_colors
import contour
import itertools

class Glyph(object):
    def __init__(self):
        globvar.glyphs.append(self)

        self.contours = []

        self.em_origin = globvar.empty_offset.copy()
        self.em_scale = 1

        self.origin_offset = globvar.empty_offset.copy()
        self.scale = 1

        self.upper_left = globvar.empty_offset.copy()
        self.lower_right = globvar.empty_offset.copy()
        self.width = 0
        self.height = 0

        return

    def destroy(self):
        for c in self.contours:
            c.destroy()
        index = globvar.glyphs.index(self)
        globvar.glyphs.pop(index)
        return

    def __len__(self):
        return len(self.contours)

    def append_contour(self, contour):
        self.contours.append(contour)
        return

    def append_contours_multi(self, contours):
        for contour in contours:
            self.append_contour(contour)
        return

    def check_all_contours_closed(self):
        for c in self.contours:
            if not c.is_closed(): return False
        return True

    def set_offset(self, offset_x, offset_y):
        self.origin_offset = np.array([offset_x, offset_y], dtype=globvar.POINT_NP_DTYPE)
        for contour in self.contours:
            contour.set_offset(offset_x, offset_y)
        return

    def set_scale(self, scale):
        self.scale = scale
        for contour in self.contours:
            contour.set_scale(self.scale)
        return

    def em_scale(self, scale):
        for contour in self.contours:
            contour.em_scale(scale)

        return

    def update_bounding_points(self):
        min_left = math.inf
        min_up = math.inf
        max_right = -math.inf
        max_down = -math.inf
        for contour in self.contours:
            up_left = contour.get_upper_left()
            down_right = contour.get_lower_right()
            min_left = min(min_left, up_left[0])
            min_up = min(min_up, up_left[1])
            max_right = max(max_right, down_right[0])
            max_down = max(max_down, down_right[1])
        self.upper_left = np.array([min_left, min_up], dtype=globvar.POINT_NP_DTYPE)
        self.lower_right = np.array([max_right, max_down], dtype=globvar.POINT_NP_DTYPE)
        self.width = self.lower_right[0] - self.upper_left[0]
        self.height = self.lower_right[1] - self.upper_left[1]
        return

    def get_bounding_box(self):
        return pygame.Rect(0, 0, self.width, self.height)

    def draw(self, surface, radius, position, color_gradient=True, width=1):
        # define a surface on which the contours can draw their fills
        bounding_box = self.get_bounding_box()
        factor = 0.2
        glyph_surface = pygame.Surface(globvar.SCREEN_DIMENSIONS)
        # glyph_surface = pygame.Surface((bounding_box.width * (1+factor), bounding_box.height * (1+factor)))
        glyph_surface.fill(custom_colors.WHITE)

        # first let them draw their respective fills
        gray_value = 0.8 * 255
        fill_color = (gray_value, gray_value, gray_value)
        for contour in self.contours:
            # TODO when refactoring to make the surface the size of the bounding box, either:
            # 1) pass an offset all the way down , -self.upper_left (kinda ugly and unmaintainable?)
            # 2) find a different way to make sure the curves draw onto this surface
            contour.draw_filled_polygon(glyph_surface, fill_color, width=width)

        # then let them draw their respective outlines
        for contour in self.contours:
            contour.draw(glyph_surface, radius, color_gradient, width=width)

        # adj_corner = [self.upper_left[0] - (bounding_box.width * factor/2), self.upper_left[1] - (bounding_box.height * factor/2)]
        # surface.blit(glyph_surface, adj_corner)
        surface.blit(glyph_surface, position)
        return

def calc_contour_score_MSE(contour1, contour2):
    return contour.find_ofer_min_mapping(contour1, contour2)

    # if len(curve1.tween_points) != len(curve2.tween_points):
    #     raise ValueError("Tried to calculate score of curves with different numbers of points")
    # # Offset the given points so that they're centered
    # c1_tween_points = (curve1.tween_points / curve1.scale) - curve1.origin_offset
    # c2_tween_points = (curve2.tween_points / curve2.scale) - curve2.origin_offset
    # return -(np.sum((c1_tween_points - c2_tween_points) ** 2) / len(c1_tween_points))

"""
Let G1 be the glyph with more contours than G2.
There isn't any limit on order like with curves, so each
contour in G1 can simply find the contour which best fits it in G2
and is of the same fill type (clockwise or CCW).
That contour will begin as a copy in G2 and become what it is in G1.
That inherently gives it a mapping.
"""
def find_glyph_null_contour_mapping(glyph1, glyph2):
    # ensure that glyph1 has no less contours than glyph2
    if len(glyph2) > len(glyph1):
        raise AttributeError("Glyph 1 needs to have at least as many curves as Glyph 2")
    g1_closure = glyph1.check_all_contours_closed()
    g2_closure = glyph2.check_all_contours_closed()

    if not (g1_closure and g2_closure):
        raise AttributeError("Both glpyhs must be closed to find an OferMin Mapping, "
                             "but: \n  ->G1's closure was " + str(g1_closure) + ", and G2's closure was " + str(
            g2_closure))

    n1 = len(glyph1)
    n2 = len(glyph2)

    # TODO implement clockwise/CCW preference in contour mapping
    # pick a mapping of glyph2's contours to glyph1's (any to any)
    g1_subsets = itertools.combinations(range(n1), n2)      # pick any n2-size subset of g1's contours
    g2_permutations = itertools.permutations(range(n2))     # try all permutations of g2's contours
    best_mapping = None
    best_score = -math.inf
    best_mapping_g1_subset = None

    # step 1:
    # try letting each contour in G2 get a unique contour in G1
    for g1_subset in g1_subsets:
        for g2_perm in g2_permutations:
            # with g1_subset and g2_perm we've decided on a potential first part of a glyph mapping
            # find the potential mapping's score by adding all of the contour mapping scores for each pair of contours
            current_mapping_set_score = 0
            current_mapping = []
            for g1_c, g2_c in zip(g1_subset, g2_perm):
                g1_contour = glyph1.contours[g1_c]
                g2_contour = glyph2.contours[g2_c]

                mapping, score = calc_contour_score_MSE(g1_contour, g2_contour)
                current_mapping_set_score += score
                # recall that we give this from g1's perspective, so a mapping pair for this 'c1' is
                # a corresponding g2 contour 'c2' and the mapping between them
                current_mapping.append([g2_c, mapping])
            if current_mapping_set_score > best_score:
                best_mapping = current_mapping
                best_score = current_mapping_set_score
                best_mapping_g1_subset = g1_subset

    # step 2:
    # we now have a mapping between n2 of the n1 contours of G1 and all of those of G2
    # we need to find the mappings for the remaining contours of G1
    unmapped_G1_contour_indices = list(set(range(n1)) - set(best_mapping_g1_subset))

    total_score = best_score
    # holds an array of the mappings of each contour in G1 to G2
    # where each mapping is of the form [index_in_G2, mapping_of_curves_for_that_contour]
    for g1_index in unmapped_G1_contour_indices:
        contour1 = glyph1.contours[g1_index]
        # find this contour's best match in glyph2
        current_best_mapping = None
        current_best_score = -math.inf
        for c2_index, contour2 in enumerate(glyph2.contours):
            # reminder: returns the best mapping of curves and score of that mapping
            mapping, mapping_score = calc_contour_score_MSE(contour1, contour2)
            if mapping_score > current_best_score:
                current_best_mapping = mapping
                current_best_score = mapping_score
        # we've now decided which of G2's contours is best
        # append contour1's mapping with it and add the mapping's score
        best_mapping.append([c2_index, current_best_mapping])
        total_score += current_best_score

    return best_mapping, total_score


def lerp_glyphs(glyph1, glyph2, mappings, t, debug_info=False):
    lerped_glpyh = Glyph()
    n1 = len(glyph1)
    n2 = len(glyph2)

    # then the rest of glyph1's contours can pick as they please
    for g1_contour_index, g1_mapping_details in enumerate(mappings):
        g2_contour_index, g2_contour_mapping = g1_mapping_details
        # print("current details:", g1_mapping_details)
        g1_contour = glyph1.contours[g1_contour_index]
        g2_contour = glyph2.contours[g2_contour_index]
        lerped_contour = contour.lerp_contours_OMin(g1_contour, g2_contour, g2_contour_mapping, t)

        # TODO set contour fill type correctly
        # TODO for now, it'll just be its original type (should still be that way later...? I think?)
        lerped_contour.fill = g1_contour.fill

        lerped_glpyh.append_contour(lerped_contour)

    return lerped_glpyh

