import numpy as np
import pygame
import math
import copy

import global_variables as globvar
import custom_math
import custom_colors
import contour
import itertools
import time

import ptext


class Glyph(object):
    def __init__(self, letter="A"):
        globvar.glyphs.append(self)

        self.contours = []

        # self.em_origin = globvar.empty_offset.copy()

        self.upper_left_world = globvar.empty_offset.copy()
        self.lower_right_world = globvar.empty_offset.copy()
        self.width = 0
        self.height = 0

        self.character_code = letter

        self.draw_surface = None
        self.draw_surface_is_updated = False
        return

    def copy(self):
        clone = copy.deepcopy(self)
        globvar.glyphs.append(clone)
        return clone

    def destroy(self):
        for c in self.contours:
            c.destroy()
        index = globvar.glyphs.index(self)
        globvar.glyphs.pop(index)
        return

    def __len__(self):
        return len(self.contours)

    def append_contour(self, contour):
        contour.character_code = self.character_code
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

    def prune_curves(self):
        for c in self.contours:
            c.prune_curves()
        return


    def worldspace_offset_by(self, offset):
        for contour in self.contours:
            contour.worldspace_offset_by(offset)

    def worldspace_scale_by(self, scale):
        for contour in self.contours:
            contour.worldspace_scale_by(scale)
        self.draw_surface_is_updated = False
        return

    def sort_contours_by_fill(self):
        direction_dictionary = {contour: contour.fill for contour in self.contours}
        direction_dictionary = custom_math.sort_dictionary(direction_dictionary, 1, reverse=True)
        self.contours = [contour for contour in direction_dictionary.keys()]
        return

    def update_bounds(self):
        min_left = math.inf
        min_up = math.inf
        max_right = -math.inf
        max_down = -math.inf
        for contour in self.contours:
            # update children contour bounds and notify their curves where the new upperleft cameraspace point is
            contour.update_bounds()

            up_left = contour.get_upper_left_world()
            down_right = contour.get_lower_right_world()
            min_left = min(min_left, up_left[0])
            min_up = min(min_up, up_left[1])
            max_right = max(max_right, down_right[0])
            max_down = max(max_down, down_right[1])
        self.upper_left_world = np.array([min_left, min_up], dtype=globvar.POINT_NP_DTYPE)
        self.lower_right_world = np.array([max_right, max_down], dtype=globvar.POINT_NP_DTYPE)
        self.width = self.lower_right_world[0] - self.upper_left_world[0]
        self.height = self.lower_right_world[1] - self.upper_left_world[1]

        for contour in self.contours:
            for curve in contour.curves:
                curve.parent_glyph_upper_left = self.get_upper_left_camera()

        return

    def reset_draw_surface(self):
        self.draw_surface_is_updated = False
        return

    def update_all_curve_points(self):
        for contour in self.contours:
            for curve in contour.curves:
                curve.update_points()
        return

    def get_upper_left_world(self):
        return self.upper_left_world

    def get_lower_right_world(self):
        return self.lower_right_world

    def get_upper_left_camera(self):
        return custom_math.world_to_cameraspace(self.upper_left_world)

    def get_lower_right_camera(self):
        return custom_math.world_to_cameraspace(self.lower_right_world)

    def get_bounding_box_world(self):
        return pygame.Rect(self.get_upper_left_world(), (self.width, self.height))

    def get_bounding_box_camera(self):
        return pygame.Rect(self.get_upper_left_camera(),
                           (self.width * globvar.CAMERA_ZOOM, self.height * globvar.CAMERA_ZOOM))

    def get_center_camera(self):
        return custom_math.world_to_cameraspace(self.get_center_world())
        # upper_left = self.get_upper_left_world()
        # world_dimensions = np.array([self.width, self.height])
        # return upper_left + (world_dimensions/2)

    """
    The center is defined to be the average of the predetermined number of points which are plotted along the curves
    """
    def get_center_world(self):

        average_point = np.array([0, 0], dtype=globvar.POINT_NP_DTYPE)

        # determine how many points each contour gets by their length
        contour_lengths = {contour_index: self.contours[contour_index].get_length_worldspace()
                           for contour_index in range(len(self))}
        total_length = sum(length for length in contour_lengths.values())
        percentages_of_total_length = {contour_index: contour_lengths[contour_index]/total_length for contour_index in contour_lengths}

        # now we need to know what overall t values and for what curves we need to get each point to be
        # the same length along the contour
        for contour_index, percentage in percentages_of_total_length.items():
            points_allotted = round(globvar.POINTS_TO_CHECK_AVERAGE_WITH * percentage)
            c = self.contours[contour_index]
            # we know how much distance should be between each point; use that
            dis_between_points = contour_lengths[contour_index] / points_allotted

            num_curves = len(c)

            # now find these points by walking along the contour and add their values to the average
            curve_index = 0
            curve_obj = c.curves[curve_index]
            point_index = 0
            point = curve_obj.worldspace_tween_points[point_index]

            prev_point = point
            between_tween_points = []
            num_tween_points = curve_obj.worldspace_tween_points.shape[0]

            point_index += 1

            for p in range(points_allotted):
                distance_remaining = dis_between_points
                # continue until you have no distance left or no curves left in the contour
                while distance_remaining > 0 and curve_index < num_curves:
                    prev_point = point
                    # walk along the next piece of the curve
                    # if you've reached the end of this curve, move to next curve
                    if point_index == num_tween_points:
                        point_index = 0
                        curve_index += 1
                        if curve_index == num_curves:
                            break
                        curve_obj = c.curves[curve_index]
                        num_tween_points = curve_obj.worldspace_tween_points.shape[0]

                    if len(between_tween_points) == 0:
                        point = curve_obj.worldspace_tween_points[point_index]
                    else:
                        point = between_tween_points.pop()

                    distance_travelled = np.linalg.norm(point - prev_point)

                    # if you don't need to use all of that distance, only use some of it
                    if distance_travelled > distance_remaining:
                        # make the current point the in between point
                        # we know how long we have to go and what line we're walking along
                        fraction_travelled = (distance_travelled - distance_remaining) / distance_travelled
                        actual_end_point = point
                        point = (point * (1 - fraction_travelled)) + (fraction_travelled * prev_point)

                        distance_travelled = distance_remaining     # for the distance_remaining -= dist_travelled calc
                        between_tween_points.append(actual_end_point)               # store this as an irregular point

                    distance_remaining -= distance_travelled

                    if len(between_tween_points) == 0:
                        point_index += 1

                # if you ran out of curves in the contour and have distance left,
                # the point naturally gets placed at the end, which is a good solution to that edge-case
                target_point = point
                average_point += target_point


        # we now have the sum of all of the surrounding points; convert to average
        average_point /= globvar.POINTS_TO_CHECK_AVERAGE_WITH

        return average_point

        # upper_left = self.get_upper_left_camera()
        # camera_dimensions = np.array([self.width, self.height]) * globvar.CAMERA_ZOOM
        # return upper_left + (camera_dimensions/2)

    def draw(self, surface, radius, width=1):
        # drawing a glyph is expensive; let's draw it once everytime the user zooms
        # when the user pans, we can just draw where we draw the glyph's *surface*
        # see the Glyph class' "worldspace_scale_by". It's there that we let the surface be None

        # TODO Could use another case; needs to be filled since the bounding box didn't change but the shape did (i.e. moving points)
        if not self.draw_surface_is_updated:
            self.draw_surface_is_updated = True
            print("Updated draw surface of ", self)

            bounding_box = self.get_bounding_box_camera()
            self.draw_surface = pygame.Surface((round(bounding_box.width) + globvar.LINE_THICKNESS,
                                                round(bounding_box.height) + globvar.LINE_THICKNESS),
                                               pygame.SRCALPHA, 32)
            self.draw_surface = self.draw_surface.convert_alpha()


            # define a surface on which the contours can draw their fills
            # bounding_box = self.get_bounding_box_camera()
            # factor = 0.2
            # glyph_surface = pygame.Surface(globvar.SCREEN_DIMENSIONS*(1+factor))
            # glyph_surface = pygame.Surface((bounding_box.width * (1+factor), bounding_box.height * (1+factor)))
            # glyph_surface.fill(custom_colors.WHITE)


            # first let them draw their respective fills
            gray_value = 0.85 * 255
            fill_color = (gray_value, gray_value, gray_value)
            for cont in self.contours:
                cont.draw_filled_polygon(self.draw_surface, fill_color, width=width)

            # then let them draw their respective outlines
            for cont in self.contours:
                cont.draw(self.draw_surface, radius, color=None, width=width)

                # if True:
                #     pnts = contour.get_points_along_from_fractions([0, 0.1, 0.5, 0.8, 0.95])
                #     num_pnts = len(pnts)
                #     for i, coords in enumerate(pnts):
                #         camera_coords = custom_math.world_to_cameraspace(coords)
                #         camera_coords = (camera_coords[0], camera_coords[1])
                #
                #         # tsurf, tpos = ptext.draw(str(adj_index), color=(0, 0, 0), center=camera_coords)
                #         # globvar.screen.blit(tsurf, tpos)
                #         col = round(255 * i / num_pnts)
                #         pygame.draw.circle(surface, (0, col, col), camera_coords, 4)

            # if True:
                #     pnts_dict = contour.get_equally_spaced_points_along(globvar.POINTS_TO_GET_CONTOUR_MAPPING_WITH,
                #                                                         return_relative_to_upper_left_curve=True)
                #     num_pnts = sum(len(pnts) for pnts in pnts_dict.values())
                #     for i, pair in enumerate(pnts_dict.items()):
                #         curve_index, pnts = pair
                #         for adj_index, coords in pnts:
                #             camera_coords = custom_math.world_to_cameraspace(coords)
                #             camera_coords = (camera_coords[0], camera_coords[1])
                #             # print(camera_coords)
                #
                #             # tsurf, tpos = ptext.draw(str(adj_index), color=(0, 0, 0), center=camera_coords)
                #             # globvar.screen.blit(tsurf, tpos)
                #             col = round(255 * adj_index / num_pnts)
                #             pygame.draw.circle(surface, (0, col, 0), camera_coords, 3)
                #
                #     # if i == round(time.time()) % len(contour):
                #     #
                #     #     break



            if True or globvar.show_extra_curve_information:
                print("SHOWING GLYPH DEBUG INFO")
                pygame.draw.circle(self.draw_surface, custom_colors.GREEN, self.get_upper_left_camera(), radius)
                # pygame.draw.circle(surface, custom_colors.GREEN, self.get_lower_right_camera(), radius)
                # pygame.draw.circle(s, custom_colors.RED, self.get_center_camera(), radius)
        surface.blit(self.draw_surface, self.get_upper_left_camera())

        # let each contour draw their points
        if globvar.show_extra_curve_information:
            for cont in self.contours:
                cont.draw_control_points(surface)
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
def find_glyph_contour_mapping(glyph1, glyph2, contour_mapping_function, debug_info=False):
    # ensure that glyph1 has no less contours than glyph2
    n1 = len(glyph1)
    n2 = len(glyph2)

    if n2 > n1:
        # raise AttributeError("Glyph 1 needs to have at least as many curves as Glyph 2")
        temp = glyph1
        glyph1 = glyph2
        glyph2 = temp

    n1 = len(glyph1)
    n2 = len(glyph2)

    g1_closure = glyph1.check_all_contours_closed()
    g2_closure = glyph2.check_all_contours_closed()

    if not (g1_closure and g2_closure):
        raise AttributeError("Both glpyhs must be closed to find a Mapping, "
                             "but: \n  ->G1's closure was " + str(g1_closure) + ", and G2's closure was " + str(
            g2_closure))


    if debug_info:
        print("Finding mapping for G1 with", n1, "contours and", str(sum(len(c) for c in glyph1.contours)), "curves")
        print("            ... and G2 with", n1, "contours and", str(sum(len(c) for c in glyph2.contours)), "curves")

    g1_center = glyph1.get_center_world()
    g2_center = glyph2.get_center_world()

    glyph1.worldspace_offset_by(-g1_center)
    glyph2.worldspace_offset_by(-g2_center)

    glyph1.update_bounds()
    glyph2.update_bounds()

    # TODO implement clockwise/CCW preference in contour mapping (currently just never picks those I think)
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
            current_mapping = {c1: None for c1 in range(n1)}
            for g1_c, g2_c in zip(g1_subset, g2_perm):
                g1_contour = glyph1.contours[g1_c]
                g2_contour = glyph2.contours[g2_c]

                mapping, score = contour_mapping_function(g1_contour, g2_contour)
                current_mapping_set_score += score
                # recall that we give this from g1's perspective, so a mapping pair for this 'c1' is
                # a corresponding g2 contour 'c2' and the mapping between them
                current_mapping[g1_c] = (g2_c, mapping)
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
        current_c2_index = -1
        current_best_score = -math.inf
        found_one_at_all = False
        for c2_index, contour2 in enumerate(glyph2.contours):
            # reminder: returns the best mapping of curves and score of that mapping
            mapping, mapping_score = contour_mapping_function(contour1, contour2)
            if mapping_score > current_best_score or not found_one_at_all:
                current_best_mapping = mapping
                current_best_score = mapping_score
                current_c2_index = c2_index
                found_one_at_all = True
        # we've now decided which of G2's contours is best
        # append contour1's mapping with it and add the mapping's score
        if current_best_mapping is None:
            print("CURRENT BEST MAPPING IS NONE! LINE 369")
        best_mapping[g1_index] = (current_c2_index, current_best_mapping)
        total_score += current_best_score

    glyph1.worldspace_offset_by(g1_center)
    glyph2.worldspace_offset_by(g2_center)

    glyph1.update_bounds()
    glyph2.update_bounds()

    return best_mapping, total_score


def lerp_glyphs(glyph1, glyph2, contour_lerping_function, mappings, t, debug_info=False):
    n1 = len(glyph1)
    n2 = len(glyph2)

    if n2 > n1:
        temp = glyph1
        glyph1 = glyph2
        glyph2 = temp

    lerped_glpyh = Glyph(glyph1.character_code)
    n1 = len(glyph1)
    n2 = len(glyph2)

    glyph1.update_bounds()
    glyph2.update_bounds()

    g1_center = glyph1.get_center_world()
    g2_center = glyph2.get_center_world()

    glyph1.worldspace_offset_by(-g1_center)
    glyph2.worldspace_offset_by(-g2_center)

    glyph1.update_bounds()
    glyph2.update_bounds()

    # mappings is a dictionary of form (contour in g1, (contour in g2, mapping from c1 -> c2))
    for g1_contour_index in mappings:
        g2_contour_index, g2_contour_mapping = mappings[g1_contour_index]
        # print("current details:", g1_mapping_details)
        g1_contour = glyph1.contours[g1_contour_index]
        g2_contour = glyph2.contours[g2_contour_index]
        lerped_contour = contour_lerping_function(g1_contour, g2_contour, g2_contour_mapping, t)

        # TODO set contour fill type correctly
        #  for now, it'll just be its original type (should still be that way later...? I think? Since
        #  we can choose the correct one in the mapping to begin with?)
        lerped_contour.fill = g1_contour.fill

        lerped_glpyh.append_contour(lerped_contour)

    glyph1.worldspace_offset_by(g1_center)
    glyph2.worldspace_offset_by(g2_center)

    glyph1.update_bounds()
    glyph2.update_bounds()

    lerped_glpyh.update_bounds()
    print("lerped glyph:", glyph1.character_code, "has", len(lerped_glpyh), "contours")
    return lerped_glpyh

def glyph_from_curves(curves):
    cont = contour.Contour()
    cont.append_curve_multi(curves)
    g = Glyph()
    g.append_contour(cont)
    return g