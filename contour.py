"""
Represents a Contour object: a path of interconnected Curve objects.
"""
import math
import random

import pygame

import curve
import global_variables as globvar
import numpy as np
import itertools
import custom_colors
import custom_math
import copy


class FILL(int):
    OUTLINE = 0
    ADD = 1
    SUBTRACT = -1


class Contour(object):
    def __init__(self):
        self.curves = []
        self.num_points = 0
        self.upper_left_world = globvar.empty_offset.copy()
        self.lower_right_world = globvar.empty_offset.copy()

        self.em_origin = globvar.empty_offset.copy()

        self.fill = FILL.ADD
        self.starting_curve = 0

        self.character_code = None

        globvar.contours.append(self)
        return

    def __len__(self):
        return len(self.curves)

    """
    For when a Contour and its curves should be garbage collected.
    Destroys curve objects and removes this Contour from the global list.
    """
    def destroy(self):
        for curve in self.curves:
            curve.destroy()
        index = globvar.contours.index(self)
        globvar.contours.pop(index)
        return

    """
    Returns a deep copy of the Contour and manually adds it to the global contour list
    """
    def copy(self):
        clone = copy.deepcopy(self)
        globvar.contours.append(clone)
        return clone

    """
    Offsets the worldspace (real) points of each curve in this Contour by offset
    """
    def worldspace_offset_by(self, offset):
        for curve in self.curves:
            curve.worldspace_offset_by(offset)
        return

    """
    Scales the worldspace (real) points of each curve in this Contour by scale
    """
    def worldspace_scale_by(self, scale):
        for curve in self.curves:
            curve.worldspace_scale_by(scale)
        return

    """
    Adds curve to this Contour's list of curves
    """
    def append_curve(self, curve):
        self.curves.append(curve)
        self.num_points += curve.tween_points.shape[0]          # for finding the average later
        return

    """
    Adds multiple curves to this Contour's list of curves
    """
    def append_curve_multi(self, curves):
        for curve in curves:
            self.append_curve(curve)
        return

    """
    Adds multiple curves to this Contour's list of curves directly from a lsit of numpy arrays
    """
    def append_curves_from_np(self, curves_point_data):
        for curve_data in curves_point_data:
            self.append_curve(curve.Bezier(curve_data))
        return


    def determine_fill_direction(self):
        curve_end_points = []
        num_curves = len(self)
        for i in range(num_curves):
            c1 = self.curves[i % num_curves]
            e1 = c1.worldspace_points[-1]
            curve_end_points.append(e1)

        points_direction = custom_math.points_clock_direction(curve_end_points)
        self.fill = FILL.ADD if points_direction == 1 else FILL.SUBTRACT

        return

    """
    Update the upper-left and lower-right (the bounding box-defining points) of this Contour
    given its curve information.
    """
    def update_bounds(self):
        left = min(c.get_upper_left_world()[0] for c in self.curves)
        top = min(c.get_upper_left_world()[1] for c in self.curves)
        right = max(c.get_lower_right_world()[0] for c in self.curves)
        bottom = max(c.get_lower_right_world()[1] for c in self.curves)
        self.upper_left_world = np.array([left, top], dtype=globvar.POINT_NP_DTYPE)
        self.lower_right_world = np.array([right, bottom], dtype=globvar.POINT_NP_DTYPE)

        return

    """
    Returns the worldspace points of each of this Contour's curves in a nicer format.
    """
    def get_str_worldspace_points(self):
        string = ""
        for curve in self.curves:
            string += str(np.round(curve.worldspace_points, 3)) + "\n"
        return string


    def get_upper_left_world(self):
        return self.upper_left_world

    def get_lower_right_world(self):
        return self.lower_right_world

    def get_upper_left_camera(self):
        return custom_math.world_to_cameraspace(self.upper_left_world)

    def get_lower_right_camera(self):
        return custom_math.world_to_cameraspace(self.lower_right_world)


    """
    Check that each curve is connected to the following
    """
    def is_closed(self):
        num_curves = len(self)
        for i in range(num_curves):
            c1 = self.curves[i % num_curves]
            c2 = self.curves[(i+1) % num_curves]
            e1 = c1.worldspace_points[-1]
            s2 = c2.worldspace_points[0]
            if not np.all(np.isclose(e1, s2)):
                return False
        return True


    """
    Returns the number of curves this contour is over the curve threshold
    and removes the shortest such curves from the contour
    """
    @DeprecationWarning
    def prune_curves(self):
        raise ValueError("THIS FUNCTION ISN'T COMPLETE! It still needs a way of reconstructing the curve from those"
                         "most crucial curves it found it wanted to keep")

        # options:
            # maybe pruning not until a number of curves is reached but removing all of the ones below a certain
                # length threshold and merging them into another bezier ("really small -> merge is less noticable)
            # maybe some sort of algorithm for first finding "similar curves"?
                # Easiest case would be turning a bunch of (nearly) straight lines into a single straight line
                    # -> try to find a line between them and check with MSE over all of those points to see if the guess is good enough?

        if len(self) <= globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD:
            return 0

        # find the "least important" curves (sort by size)
        curve_lengths = {index: c.get_length() for index, c in enumerate(self.curves)}
        curve_lengths = {k: v for k, v in sorted(curve_lengths.items(), key=lambda item: item[1], reverse=True)}

        print(curve_lengths)

        # pick the highest ones to keep
        kept_indices = [index for i, index in enumerate(curve_lengths.keys()) if i < globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD]
        kept_indices.sort()
        print(kept_indices, len(kept_indices))
        self.num_points = 0
        reconstructed_curves = []
        for i in kept_indices:
            c = self.curves[i]
            reconstructed_curves.append(c)

        self.curves = reconstructed_curves

        # if closure was ruined, add a closure line
        if not self.is_closed():
            print("!!")

        return globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD - len(self)


    """
    Returns the worldspace (real) length of the entire contour
    """
    def get_length_worldspace(self):
        return sum(c.get_length_world() for c in self.curves)


    @DeprecationWarning
    def update_curve_center_relative_angles(self):
        center = self.get_center()
        for curve in self.curves:
            # offset the curve's center to "(0, 0)" by subtracting 'center'
            # note that we work in degrees
            adjusted_point = curve.average_point - center
            cur_curve_rel_angle = np.arctan(adjusted_point[1] / adjusted_point[0]) * 180 / np.pi
            curve.current_contour_relative_angle = cur_curve_rel_angle
            # print("rel angle:", cur_curve_rel_angle)
        return


    def get_anchor_world(self):
        top_left = self.get_upper_left_world()
        down_right = self.get_lower_right_world()
        top_right = np.array([down_right[0], top_left[1]])
        down_left = np.array([top_left[0], down_right[1]])
        if self.character_code in globvar.up_lefters:
            return top_left
        elif self.character_code in globvar.up_righters:
            return top_right
        elif self.character_code in globvar.down_righters:
            return down_right
        elif self.character_code in globvar.down_lefters:
            return down_left
        else:   # commas and stuff for now
            return down_right

    """
    Returns a list of num_points points spaced equally along the contour.
    """
    def get_equally_spaced_points_along(self, num_points,
                                        return_relative_to_upper_left_curve=False):

        resultant_points = []
        curves_and_points = {index: [] for index in range(len(self))}

        contour_length_worldspace = self.get_length_worldspace()

        # we know how much distance should be between each point
        dis_between_points = contour_length_worldspace / num_points

        num_curves = len(self)
        self_upper_left = self.get_upper_left_world()

        # now find these points by walking along the contour and add their values to the average
        curve_index = 0
        curve_obj = self.curves[curve_index]
        point_index = 0
        point = curve_obj.worldspace_tween_points[point_index]

        prev_point = point
        between_tween_points = []
        num_tween_points = curve_obj.worldspace_tween_points.shape[0]

        curve_with_upper_left_point = 0
        upper_left_point = point
        upper_left_point_len = np.linalg.norm(point - self_upper_left)

        point_index += 1

        for p in range(num_points):
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
                    curve_obj = self.curves[curve_index]
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

                    distance_travelled = distance_remaining  # for the distance_remaining -= dist_travelled calc
                    between_tween_points.append(actual_end_point)  # store this as an irregular point


                distance_remaining -= distance_travelled

                if len(between_tween_points) == 0:
                    point_index += 1


            # get the distance from the point to the contour's upper left corner
            point_len = np.linalg.norm(point - self_upper_left)
            if point_len < upper_left_point_len:
                upper_left_point = point
                upper_left_point_len = point_len
                curve_with_upper_left_point = curve_index % num_curves

            # if globvar.show_extra_curve_information:
            #     print(upper_left_point)

            # if you ran out of curves in the contour and have distance left,
            # the point naturally gets placed at the end, which is a good solution to that edge-case
            resultant_points.append(point)
            if return_relative_to_upper_left_curve:
                curves_and_points[(curve_index % num_curves)].append((p, point))

            # pygame.draw.circle(globvar.screen, (0, 255 * p / num_points, 0),
            #                    custom_math.world_to_cameraspace(point),
            #                    3)


        result = resultant_points

        # anchor = np.array(self.get_lower_right_world()[0], self.get_upper_left_world()[1])
        anchor = self.get_anchor_world()
        if return_relative_to_upper_left_curve:

            # get the curve with smallest min squared error to upper left point
            lowest_MSE = math.inf
            upper_leftest_curve = 0
            for curve_index in range(num_curves):
                pnts = [coords for index, coords in curves_and_points[curve_index]]
                if len(pnts) > 0:
                    MSE = sum(np.linalg.norm(anchor - coords) for coords in pnts) / len(pnts)
                    if MSE < lowest_MSE:
                        upper_leftest_curve = curve_index
                        lowest_MSE = MSE

        #     print(upper_leftest_curve)
        #     print(curves_and_points)
        #     print("starting curve:", self.starting_curve)
        #     print("now both", curves_and_points[self.starting_curve])
            first_index_in_up_left, first_coords_in_up_left = curves_and_points[upper_leftest_curve][0]

            adjusted_points_in_curve_groups = {}
            for pair in curves_and_points.items():
                curve_index, points_group = pair
                adjusted_index = (curve_index - curve_with_upper_left_point) % num_curves
                adjusted_points_group = [((p_index - first_index_in_up_left) % num_points, p_coords) for p_index, p_coords in
                                         points_group]
                adjusted_points_in_curve_groups[adjusted_index] = adjusted_points_group

            result = adjusted_points_in_curve_groups

        return result


    """
    Return a list of points along this contour corresponding to the list of fractions given
    """
    def get_points_along_from_fractions(self, fractions_along_of_points,
                                        return_with_curves=False):


        fractions_along_of_points.sort()    # so we can traverse them in order
        fractions_along_of_points.append(1) # so the last difference between fractions gets a chance to be processed
        num_points = len(fractions_along_of_points)

        resultant_points = []
        curves_and_points = {index: [] for index in range(len(self))}

        contour_length_worldspace = self.get_length_worldspace()

        num_curves = len(self)
        self_upper_left = self.get_upper_left_world()

        # now find these points by walking along the contour and add their values to the average
        curve_index = 0
        curve_obj = self.curves[curve_index]
        point_index = 0
        point = curve_obj.worldspace_tween_points[point_index]

        prev_point = point
        between_tween_points = []
        num_tween_points = curve_obj.worldspace_tween_points.shape[0]

        point_index += 1

        # we want to traverse the contour in order, so the fractions_along_of_points
        # list is sorted; this allows us to consider the distance_remaining to
        # be the difference in the previous total distance and the current
        distance_remaining = contour_length_worldspace * fractions_along_of_points[0]
        distance_to_run = distance_remaining
        prev_distance_to_run = 0
        for p, fraction_along in enumerate(fractions_along_of_points):
            distance_remaining = distance_to_run - prev_distance_to_run

            # update for next loop
            prev_distance_to_run = distance_to_run
            distance_to_run = contour_length_worldspace * fraction_along

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
                    curve_obj = self.curves[curve_index]
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

                    distance_travelled = distance_remaining  # for the distance_remaining -= dist_travelled calc
                    between_tween_points.append(actual_end_point)  # store this as an irregular point


                distance_remaining -= distance_travelled

                if len(between_tween_points) == 0:
                    point_index += 1


            # get the distance from
            resultant_points.append(point)
            if return_with_curves:
                curves_and_points[(curve_index % num_curves)].append((p, point))

        result = resultant_points

        if return_with_curves:
            result = curves_and_points

        return result


    # ==== Drawing ====

    """
    Draws the outline of this Contour onto the given surface with a given color.
    """
    def draw(self, surface, radius, color=None, width=1):
        # draw curves in colors corresponding to their order in this contour
        no_input_color = color is None
        for i, curve in enumerate(self.curves):
            if no_input_color:
                final_color = [custom_colors.LT_GRAY,
                                custom_colors.mix_color(custom_colors.GREEN, custom_colors.RED, (i+1)/(len(self)+1)),
                                custom_colors.GRAY]
            else:
                final_color = [color, color, color]
            curve.draw(surface, radius, final_color, width=width)
        return

    def draw_curve_bound_points(self, surface, radius):
        # draw debug information (center, etc.)
        if globvar.show_extra_curve_information:
            pygame.draw.circle(surface, custom_colors.RED, self.get_upper_left_camera(), radius * 0.75)
            pygame.draw.circle(surface, custom_colors.RED, self.get_lower_right_camera(), radius * 0.75)

    """
    Draws the control points defining each of this Contour's curve objects.
    For annoying cutoff reasons, isn't included in a glyph's draw-caching and is instead drawn straight to the screen.
    """
    def draw_control_points(self, surface):
        for curve in self.curves:
            curve.draw_control_points(globvar.screen, radius=globvar.POINT_DRAW_RADIUS)
        return

    """
    Draws a polygon in the space defined by the contour onto the specified surface.
    flush_with_origin keeps the polygon flush with a parent glyph's bounding box and is used
    when using draw-caching, wherein the glyph only redraws itself when changes are made.
    Setting it to false could be useful if it's easier to draw the contour independently of its parent glyph.
    
    Won't throw an error if the contour isn't closed. 
    """
    def draw_filled_polygon(self, surface, fill_color, width=1, flush_with_origin=True):
        # get all of the points of this contour's lines
        all_tween_points = []
        for curve in self.curves:
            reference_points = curve.unoffset_tween_points if flush_with_origin else curve.tween_points
            for tween_point in reference_points:
                all_tween_points.append(tween_point)

        fill_flag = self.fill
        if self.fill == FILL.ADD:
            fill_flag = 0
        if self.fill == FILL.SUBTRACT:
            fill_flag = 0
            fill_color = custom_colors.WHITE
        if self.fill == FILL.OUTLINE:
            fill_flag = width

        contour_bounding_box = pygame.draw.polygon(surface, fill_color, all_tween_points, width=round(fill_flag))
        return contour_bounding_box


"""
An attempt at a more generic scoring function. Likely has logical errors...
"""
@DeprecationWarning
def calc_score(curve1, curve2):
    # Attribute weights
    # length, angle between endpoints, angle from contour center
    weights = np.array([0.15, 0.3, 0.55])

    delta_length = abs(curve1.get_length() - curve2.get_length())
    delta_endpoint_angle = abs(curve1.get_angle() - curve2.get_angle())
    delta_relative_angle = abs(curve1.current_contour_relative_angle - curve2.current_contour_relative_angle)
    deltas = np.array([delta_length, delta_endpoint_angle, delta_relative_angle])

    # print("deltas", deltas)
    score = np.inner(weights, deltas)
    # print("score", score)
    return score


"""
Given two curve objects, calculates their Mean-Squared Error.
"""
@DeprecationWarning
def calc_curve_score_MSE(curve1, curve2):
    if len(curve1.tween_points) != len(curve2.tween_points):
        raise ValueError("Tried to calculate score of curves with different numbers of points")

    # Offset the given points so that they're centered
    c1_tween_points = custom_math.camera_to_worldspace(curve1.tween_points)
    c2_tween_points = custom_math.camera_to_worldspace(curve2.tween_points)
    return -(np.sum((c1_tween_points - c2_tween_points) ** 2) / len(c1_tween_points))


"""
A remnant of an attempt at using a greedy variant of the Reduction method.
Does a weird sort of binary search to find the first operand of a choose function that will maximize without
going over some threshold value.
"""
@DeprecationWarning
def maximal_curves_for_threshold(n1, n2, threshold):
    # basically performs a binary search on the scale of the choose function
    prev_ncr = custom_math.ncr(n1, n2)
    cur_top_val = round(0.5 * (n1 + 2))       # start from the average value
    largest_valid_top_val = n2
    visited_top_values = []
    while prev_ncr > 1:
        prev_ncr = custom_math.ncr(cur_top_val, n2)
        if prev_ncr > threshold:        # go lower
            cur_top_val = round(0.5 * (cur_top_val + n2))
        else:                           # go higher
            largest_valid_top_val = max(largest_valid_top_val, cur_top_val)
            cur_top_val = round(0.5 * (n1 + cur_top_val))
        if cur_top_val in visited_top_values:
            return largest_valid_top_val
        visited_top_values.append(cur_top_val)


    # should never reach here
    return n2



# Finding mappings =============================================

"""
Performs basic checks on the contours that must occur regardless of mapping function.
"""
def contour_mapping_preprocess(contour1: Contour, contour2: Contour, contour_mapping_function):
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp


    if contour1.fill != contour2.fill:
        print("Fills don't match! Beware of strange results!")

    return contour_mapping_function(contour1, contour2)

"""
For two contours such that |C1| >= |C2|:
Uses the "Reduction" method. Currently exhaustive, simply searching to find which subset of |C2| curves of C1
will best fit C2's curves and ignores the rest.

Returns a mapping where each pair is of the form (C1 curve, C2 curve) and a score - the higher the better.
"""
@DeprecationWarning
def find_reduction_mapping(contour1: Contour, contour2: Contour):
    n1 = len(contour1)
    n2 = len(contour2)

    index_subsets = itertools.combinations(range(n1), n2)  # returns all sets of indices in [0, n-1] of size n2
    range_n2 = list(range(n2))

    for indices in index_subsets:
        # iterate through all offsets
        for offset in range_n2:
            # use the offset by iterating through all placements
            # build a score for this mapping (based on both the indices and offset currently being tested)
            current_mapping_score = 0
            current_mapping_dict = {c1: None for c1 in range(n1)}
            for c1_index, c2_index in zip(indices, range_n2):
                c1_curve = contour1.curves[c1_index]
                c2_curve = contour2.curves[(c2_index + offset) % n2]
                current_mapping_score += calc_curve_score_MSE(c1_curve, c2_curve)
                current_mapping_dict[c1_index] = (c2_index + offset) % n2
            if current_mapping_score > best_score:
                best_score = current_mapping_score
                best_mapping = current_mapping_dict

    return best_mapping, best_score


"""
Uses the "Pillow Projection" method.
Projects a predetermined number of points onto each contour, uses an anchor point relative to each contour to determine
which of those points should be that contour's "point 0". Matches points of the same index on each contour.

Returns a mapping where each pair is of the form (point on C1, point on C2), and a score - the higher the better.
"""
def find_pillow_projection_mapping(contour1: Contour, contour2: Contour):
    n1 = len(contour1)
    n2 = len(contour2)

    num_sample_points = globvar.POINTS_TO_GET_CONTOUR_MAPPING_WITH
    contours = [contour1, contour2]
    equidis_point_lists = []
    starting_points = []
    for contour in contours:
        equidis_points = contour.get_equally_spaced_points_along(num_sample_points)

        # @TODO Reassess: how best to pick anchor?
        #   maybe change this to whichever anchor which minimizes MSE to each contour's center?
        anchor = contour.get_upper_left_world()

        # find the point closest to the anchor
        closest_point = custom_math.find_point_closest_to_anchor(anchor, equidis_points, return_point_index=True)
        closest_point_index = closest_point[0]
        starting_points.append(closest_point)

        # find best rotational offset by assigning each point an index based on that which
        # is closest to an arbitrary anchor; offset the points in each such that they start from closest point
        equidis_points = {(index - closest_point_index) % num_sample_points: coords
                          for index, coords in enumerate(equidis_points)}
        equidis_points = custom_math.sort_dictionary(equidis_points, by_key_or_value=0)
        equidis_points = equidis_points.values()

        equidis_point_lists.append(equidis_points)

    mapping = {i: (p1, p2) for i, (p1, p2) in enumerate(zip(equidis_point_lists[0], equidis_point_lists[1]))}
    score = 0
    for p1, p2 in mapping.values():
        score += np.sum((p1 - p2) ** 2)

    score *= -1 / num_sample_points         # lower MSE is better so we multiply by -1

    return mapping, score


# @TODO Implemented hastily. Needs rewrite.
"""
Finds a "Relative Projection" mapping. That is, we attempt to project the Bezier endpoints of each contour onto
the other. This is done based on some relative anchor point to determine the fractional values of how far along
the contour a given endpoint is. 

Returns a mapping in the form of {{(contour1, contour2), rest_of_mapping_between_their_curves}}, score
"""
def find_relative_projection_mapping(contour1: Contour, contour2: Contour):
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp

    n1 = len(contour1)
    n2 = len(contour2)

    # split each curve into many points along each; these will work as fine-grain units
    num_sample_points = globvar.POINTS_TO_GET_CONTOUR_MAPPING_WITH
    contours = [contour1, contour2]
    equidis_point_lists = []
    starting_points = []
    control_point_percentages_dicts = []
    control_point_percentages_lists = []
    for contour in contours:
        equidis_points = contour.get_equally_spaced_points_along(num_sample_points)

        # anchor = contour.get_upper_left_world()         # TODO maybe change this to whichever anchor which minimizes MSE to both contours' centers?
        anchor = contour.get_anchor_world()

        # find the point closest to the anchor
        start_point = custom_math.find_point_closest_to_anchor(anchor, equidis_points, return_point_index=True)
        start_point_index = start_point[0]
        starting_points.append(start_point)

        # find best rotational offset by assigning each point an index based on that which
        # is closest to an arbitrary anchor; offset the points in each such that they start from closest point
        equidis_points = {(index - start_point_index) % num_sample_points: coords
                          for index, coords in enumerate(equidis_points)}
        equidis_points = custom_math.sort_dictionary(equidis_points, by_key_or_value=0)

        # equidis_points is now the points sorted so that the 0th entry is that point closest to anchor
        # and the rest follow cyclically around the contour
        equidis_points = list(equidis_points.values())

        equidis_point_lists.append(equidis_points)
        globvar.random_debug_points.append(equidis_points[0])

        # we now need to find the approximate locations of this contour's control points compared to these unit points
        control_point_percentages = {curve: [] for curve in contour.curves}
        control_point_percentages_all = []
        for curve in contour.curves:
            for control_point in [curve.worldspace_points[0], curve.worldspace_points[-1]]:
                # for now we'd like to only consider control points on the curve - those which partition it
                # if #curve.contains_point(control_point): # TODO used to check if curves where on curve but now just using endpoints
                # on_curve_control_points.append(control_point)
                point_index, approximate_coords = custom_math.find_point_closest_to_anchor(control_point,
                                                                                           equidis_points,
                                                                                           return_point_index=True)
                # convert to a fraction along from the start point determined before
                fraction_around = point_index / num_sample_points
                control_point_percentages[curve].append(fraction_around)
                control_point_percentages_all.append(fraction_around)


        # on_curve_control_point_lists.append(on_curve_control_points)
        control_point_percentages_dicts.append(control_point_percentages)
        control_point_percentages_lists.append(control_point_percentages_all)


    # ==== END PREPROCESSING SECTION =====
    #
    #
    #     # c2_coords_on_c1 = contour1.get_points_along_from_fractions(c2_fractions_around)
    #     # c1_coords_on_c2 = contour2.get_points_along_from_fractions(c1_fractions_around)
    #
    #
    # # generate two new contours with the same number of control points:
    # # C1_, with C2's points marked on it and split accordingly,
    # # and C2_ with C1's points marked on it and split accordingly
    #     # c1_on_curve_points, c2_on_curve_points = on_curve_control_point_lists
    #
    # # for c1, for example, the dicts's points should look like:
    # # {(c1 curve object): [c2_points that fall between its endpoints]}
    # # where each point is of the form (fraction units, coords)
    # # and both the keys and their lists should be in order of fraction along
    #     # marked_c1_coords = c2_coords_on_c1 + c1_on_curve_points     # c1, marked with its own points and c2's
    #     # marked_c2_points = c1_coords_on_c2 + c2_on_curve_points     # c2, marked with its own points and c1's
    #
    #     # num_marked = len(marked_c1_coords)
    #     # num_marked2 = len(marked_c2_points)
    #     # print("do they have the same number of coords? they should!", num_marked == num_marked2)
    # #
    # # cur_c1_point = marked_c1_coords[0]
    # # for c1_i in range(num_marked + 1):
    # #     cur_is_c1, cur_coords = marked_c1_coords[c1_i % num_marked]
    # #     next_is_c1, next_coords = marked_c1_coords[(c1_i + 1) % num_marked]
    # #
    # # start building marked_c1_coords in the required form; begins from any point since both will have all points
    #
    # # ===== BEGIN CONTOUR SPLITTING =====
    # # for each contour we find where its on-curve control points fall, fraction-wise, along the other one
    # c1_control_points_fractions, c2_control_points_fractions = control_point_percentages_dicts
    # c1_control_point_fractions_all, c2_control_point_fractions_all = control_point_percentages_lists
    #
    # # # these are lists of points sorted by adjusted index (starting from point closest to anchor)
    # # points_along_c1, points_along_c2 = equidis_point_lists

    split_contours = []
    mapping = {}        # for each pair of percentages bounding a curve store that curve

    def add_to_mapping(endpoint1_percent, endpoint2_percent, subcurve):
        endpoint1_percent = round(endpoint1_percent, 7)
        endpoint2_percent = round(endpoint2_percent, 7)

        # allowing the values to be over 1 is great for sorting
        # when it comes to grouping them together, though, different contours
        # may choose to add 1 for different fraction-values;
        # thus we normalize it all back to the same unit space when adding in the mapping
        if endpoint1_percent > 1:
            endpoint1_percent -= 1
        if endpoint2_percent > 1:
            endpoint2_percent -= 1

        if (endpoint1_percent, endpoint2_percent) not in mapping.keys():
            # print("!", endpoint1_percent, endpoint2_percent)
            mapping[(endpoint1_percent, endpoint2_percent)] = []
            # print(mapping[(endpoint1_percent, endpoint2_percent)])


        # print((endpoint1_percent, endpoint2_percent) not in mapping.keys())
        mapping[(endpoint1_percent, endpoint2_percent)].append(subcurve)
        return

    for contour_index, contour_being_split in enumerate([contour1, contour2]):
        # "my" refers to the contour currently being split
        # "their" or "other" refers to the contour whose point-fraction-values are being references
        # e.g. if contour_being_split = contour1 then my_control_points_fractions = c1_control_points_fractions
        my_control_points_fractions = control_point_percentages_dicts[contour_index]
        their_control_point_fractions_all = control_point_percentages_lists[1 - contour_index]

        # TODO === TESTING ===
        # testing that the points being found, in fractional form, are reflected correctly;
        # take each OTHER contour's fraction-unit points (FUPs) and place them onto this one,
        # in index order so they can be differentiated later
        both_contours = [contour1, contour2]
        others_marked_points = both_contours[contour_index].get_points_along_from_fractions(their_control_point_fractions_all)
        globvar.marking_test_points_lists.append(others_marked_points)
        # TODO === TESTING ===

        split_contour = Contour()
        split_contours.append(split_contour)

        for curve in contour_being_split.curves:
            # interval_start_point = c1_curve.worldspace_points[0]
            # interval_end_point = c1_curve.worldspace_points[-1]
            #
            # interval_start_point_index, start_approximate_coords = custom_math.find_point_closest_to_anchor(
            #                                                                             interval_start_point,
            #                                                                             points_along_c1,
            #                                                                             return_point_index=True)
            # interval_start_fraction = interval_start_point_index / num_sample_points
            #
            # interval_end_point_index, end_approximate_coords = custom_math.find_point_closest_to_anchor(
            #                                                                             interval_end_point,
            #                                                                             points_along_c1,
            #                                                                             return_point_index=True)
            # interval_end_fraction = interval_end_point_index / num_sample_points
            interval_start_fraction, interval_end_fraction = my_control_points_fractions[curve]

            # start the splitting process by just taking the entire current curve
            remaining_interval_bezier = curve
            # does this curve interval start before 0%/100% and end after it (bridges the loop around?)
            curve_loops_around = interval_start_fraction > interval_end_fraction
            if curve_loops_around:
                print("")
                print("loops around with start and end", interval_start_fraction, interval_end_fraction)
                interval_end_fraction += 1

            # get all of the points on c2 that are between these control points
            contained_fractions_from_other = []
            for fraction_value in their_control_point_fractions_all:
                in_interval_regular = interval_start_fraction < fraction_value < interval_end_fraction
                in_interval_other_side = False#curve_loops_around and fraction_value < interval_end_fraction
                if in_interval_regular or in_interval_other_side:
                    if fraction_value < interval_start_fraction and curve_loops_around:    # if this is a point after the contour's start point we found earlier
                        contained_fractions_from_other.append(fraction_value + 1)
                    else:
                        contained_fractions_from_other.append(fraction_value)

            # sort the contained fractions so that the points are used to split the bezier in order
            contained_fractions_from_other.sort()

            # begin actually making the new curves from the current contour's current endpoints and the newly found
            # midway points
            for fraction_value in contained_fractions_from_other:
                # continue splitting until you have no more c2 points between this curve's endpoints
                interval_fraction_range = interval_end_fraction - interval_start_fraction
                if interval_fraction_range > 0 and fraction_value - interval_start_fraction > 0:
                    percent_of_remaining_bezier = (fraction_value - interval_start_fraction) / interval_fraction_range

                    bezier1, bezier2 = remaining_interval_bezier.de_casteljau(percent_of_remaining_bezier)
                    split_contour.append_curve(bezier1)

                    add_to_mapping(interval_start_fraction, fraction_value, bezier1)

                    remaining_interval_bezier = bezier2

                    # move the interval's new 'start' to where this one just made the split
                    interval_start_fraction = fraction_value

            # END CURVE SPLITTING LOOP FOR ONE CURVE - AFTER APPENDING THIS LAST SUBCURVE PROCEED TO SPLIT NEXT CURVE

            # append the last curve created - it isn't included in the loop above
            split_contour.append_curve(remaining_interval_bezier)
            # interval_start_fraction is now the last fraction_value of the loop
            add_to_mapping(interval_start_fraction, interval_end_fraction, remaining_interval_bezier)

        split_contour.update_bounds()


    score = 0
    num_curve_pairs = len(mapping)
    for endpoint_percentage_pair in mapping.keys():
        urg = len(mapping[endpoint_percentage_pair])
        if urg < 2:
            print("LOOK AT LINE 781!!! GOT MAPPING OF LENGTH:", urg)
            for key in mapping:
                print(key, mapping[key])
        # TODO REMOVE THIS OVERRIDE REMOVE IT REMOVE IT
        if urg > 1:
            bezier1, bezier2 = mapping[endpoint_percentage_pair]
            score += calc_curve_score_MSE(bezier1, bezier2)

    score *= -1 / num_curve_pairs         # lower MSE is better

    if contour1.fill != contour2.fill:
        print("Fills don't match but were the only options! Beware of strange results!")

    return mapping, score


# ==== Interpolation functions ====
def contour_lerping_preprocess(contour1: Contour, contour2: Contour, mapping: dict, t, lerping_function):

    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp
        t = 1-t

    return lerping_function(contour1, contour2, mapping, t)


# @TODO Warning! Searches exhaustively! Prohibitively slow!
@DeprecationWarning
def lerp_contours_reduction(contour1: Contour, contour2: Contour, mapping: dict, t, debug_info=False):
    # we know we need n1 curves in total
    # we also know which indices of c1 were chosen
    # therefore for any index not after the expected one (i.e. there is a gap between one index and the next)
    # we fill with a "zero curve"
    # mapping is a dictionary where each entry is of the form (c1, c2) and if that c1 curve has no mapping, (c1, None)
    # make the same switch as the mapping did; from here, everything should be the same

    n1 = len(contour1)
    n2 = len(contour2)

    lerped_contour = Contour()
    # first find where we start building from

    first_c1_index = None
    first_c2_index = None
    for pair in mapping.items():
        first_c1_index, first_c2_index = pair
        if first_c2_index is not None:
            break

    last_endpoint = contour2.curves[first_c2_index].worldspace_points[0]
    for pair_index in range(n1):
        # get the pair in the current cyclic index
        c1_index = (pair_index + first_c1_index) % n1
        c2_index = mapping[c1_index]
        if debug_info:
            print("adding curve #", c1_index, ", a mapping between C2's", c2_index, "and C1's",
                  c1_index)
        if c2_index is None:            # make a null curve
            # interpolate between the current c1_curve and the "zero curve" which
            # consists of four points bunched together at c2's last point
            contour1_curve_true_points = contour1.curves[c1_index].worldspace_points
            degree_polynomial = contour1_curve_true_points.shape[0]
            lerped_zero_curve_points = custom_math.interpolate_np(
                contour1_curve_true_points,
                np.array([last_endpoint] * degree_polynomial),
                t)
            lerped_contour.append_curve(curve.Bezier(lerped_zero_curve_points))
        else:                           # make a curve interpolated between the c1_index and c2_index curves
            # now add the curve you've been trying to give me (we're all caught up to the expected index)
            # by interpolating between the current c2 curve and the corresponding c1 curve
            lerped_points = custom_math.interpolate_np(contour1.curves[c1_index].worldspace_points,
                                                       contour2.curves[c2_index].worldspace_points,
                                                       t)
            last_endpoint = contour2.curves[c2_index].worldspace_points[-1]
            lerped_contour.append_curve(curve.Bezier(lerped_points))
    return lerped_contour


def lerp_contours_pillow_proj(contour1, contour2, point_mapping, t):

    n1 = len(contour1)
    n2 = len(contour2)

    lerped_contour = Contour()

    num_pairs = len(list(point_mapping.keys()))
    lerped_points = []
    for pair in point_mapping.values():
        p1, p2 = pair
        lerped_points.append(custom_math.interpolate_np(p1, p2, t))

    lerped_contour = connect_points_with_cubic_beizers(lerped_points)

    # Old but keeping this here so the logic is visible
    # # just make a new contour from quadratic beziers connecting each point linearly
    # prev_point = lerped_points[0]
    # point = prev_point
    # for i in range(num_pairs + 1):
    #     cyclic_index = i % num_pairs
    #     prev_point = point
    #     point = lerped_points[cyclic_index]
    #     line_points = np.array([point, custom_math.interpolate_np(point, prev_point, 0.5), prev_point])
    #     new_curve = curve.Bezier(line_points)
    #     lerped_contour.append_curve(new_curve)

    lerped_contour.update_bounds()
    return lerped_contour


def lerp_contours_relative_proj(contour1, contour2, curve_mapping, t):
    n1 = len(contour1)
    n2 = len(contour2)

    lerped_contour = Contour()

    # in this method we simply need to lerp between the curves at each endpoint-pair-key
    for curve_pair in curve_mapping.values():
        # TODO REMOVE REMOVE REMOVE THIS MANUAL OVERRIDE!!
        if len(curve_pair) > 1:
            bezier1, bezier2 = curve_pair

            lerp_curve = curve.Bezier(custom_math.interpolate_np(bezier1.worldspace_points,
                                                                 bezier2.worldspace_points,
                                                                 t))
            lerped_contour.append_curve(lerp_curve)

    lerped_contour.update_bounds()
    return lerped_contour



# ==== Basic shape contours ====
def get_unit_circle_contour():
    circle_const = globvar.CIRCLE_CONST
    circle_a1 = np.array([[0, 0],
                          [circle_const, 0],
                          [1, 1 - circle_const],
                          [1, 1]])
    circle_a2 = np.array([[1, 1],
                          [1, 1 + circle_const],
                          [circle_const, 2],
                          [0, 2]])
    circle_a3 = np.array([[0, 2],
                          [-circle_const, 2],
                          [-1, 1 + circle_const],
                          [-1, 1]])
    circle_a4 = np.array([[-1, 1],
                          [-1, 1 - circle_const],
                          [-circle_const, 0],
                          [0, 0]])

    circle = Contour()
    circle.append_curves_from_np([circle_a1, circle_a2, circle_a3, circle_a4])
    return circle

def get_unit_polygon_contour(sides, scale, angle_offset=0):
    if sides < 3:
        raise ValueError("Polygon Contour must have at least 3 sides.")
    vertices = [np.array([np.cos(angle_offset-(np.pi/2) + (i*2*np.pi/sides)),
                          np.sin(angle_offset-(np.pi/2) + (i*2*np.pi/sides))])*scale for i in range(sides)]

    polygon = Contour()

    for i in range(sides):
        vertex = vertices[i]
        next_vertex = vertices[(i+1) % sides]

        side = curve.Bezier(np.array([vertex, custom_math.interpolate_np(vertex, next_vertex, 0.5), next_vertex]))
        polygon.append_curve(side)
        print(vertex, next_vertex)

    return polygon


"""
From https://towardsdatascience.com/b%C3%A9zier-interpolation-8033e9a262c2 
"""
# find the a & b points
def get_bezier_coef(input_points):
    points = np.vstack([input_points, input_points[0]]) # so that we can close the contour later
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

def connect_points_with_cubic_beizers(points):
    A, B = get_bezier_coef(points)

    cont = Contour()

    num_points = len(points)
    for i in range(num_points-1):
        connecting_curve = curve.Bezier(np.array([points[i], A[i], B[i], points[i+1]]))
        cont.append_curve(connecting_curve)

    connecting_curve = curve.Bezier(np.array([points[-1], A[-1], B[-1], points[0]]))
    cont.append_curve(connecting_curve)

    return cont


# @TODO Worked out on paper. Yet to be implemented.
"""
A custom point-to-Bezier smoothing function. Given n points, returns ceiling(n * 2/3) cubic bezier curves
which smoothly (matching first derivative at seam points) connect the points.
"""
def quad_bezier_contour_from_points(points):
    raise ValueError("Yet unimplemented!")
    cont = Contour()

    # go through the points in sets of 3 if possible
    num_points = points.shape[0]
    num_sets_of_three = num_points // 2     # 2 connections between 3 points so divide by 2

    sets = []
    p1 = points[0]
    p2 = points[1]
    p3 = None

    # # begin with curve unconstrained by continuity; we'll fix it at the end
    # if num_points > 2:
    #     p3 = points[3]
    #
    #     # determine the middle control_point arbitarily with t = 0.5
    #     p_control = (2*p2) - (0.5*(p1+p3))
    #
    #     # make a quadratic bezier curve with the current points
    #     cur_curve = curve.Bezier([p1, p_control, p3])

    # note that the first curve will be wrong initially; we'll fix it at the end

    # 3-size sets
    for i in range(num_sets_of_three):
        p2 = points[(3*i)+1]
        p3 = points[(3*i)+2]

        # determine the middle control point from t = 0.5 and the previous center control_point
        p_control = (2 * p2) - (0.5 * (p1 + p3))

        # make a quadratic bezier curve with the current points
        cur_curve = curve.Bezier([p1, p_control, p3])

        # make a quadratic bezier curve with the current points
        cur_curve = curve.Bezier()


        sets.append([p1, p2, p3])

        p1 = p3

    # possible last set in case of odd number of points
    p2 = points
    sets.append([])


    return cont
