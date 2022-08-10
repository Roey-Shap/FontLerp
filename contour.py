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

    def destroy(self):
        for curve in self.curves:
            curve.destroy()
        index = globvar.contours.index(self)
        globvar.contours.pop(index)
        return

    def copy(self):
        clone = copy.deepcopy(self)
        globvar.contours.append(clone)
        return clone

    def worldspace_offset_by(self, offset):
        for curve in self.curves:
            curve.worldspace_offset_by(offset)
        return

    def worldspace_scale_by(self, scale):
        for curve in self.curves:
            curve.worldspace_scale_by(scale)
        return

    def append_curve(self, curve):
        self.curves.append(curve)
        self.num_points += curve.tween_points.shape[0]          # for finding the average later
        return

    def append_curve_multi(self, curves):
        for curve in curves:
            self.append_curve(curve)
        return

    def append_curves_from_np(self, curves_point_data):
        for curve_data in curves_point_data:
            self.append_curve(curve.Bezier(curve_data))
        return


    def update_bounds(self):
        left = min(c.get_upper_left_world()[0] for c in self.curves)
        top = min(c.get_upper_left_world()[1] for c in self.curves)
        right = max(c.get_lower_right_world()[0] for c in self.curves)
        bottom = max(c.get_lower_right_world()[1] for c in self.curves)
        self.upper_left_world = np.array([left, top], dtype=globvar.POINT_NP_DTYPE)
        self.lower_right_world = np.array([right, bottom], dtype=globvar.POINT_NP_DTYPE)

        curve_end_points = []
        num_curves = len(self)
        for i in range(num_curves):
            c1 = self.curves[i % num_curves]
            e1 = c1.worldspace_points[-1]
            curve_end_points.append(e1)

        points_direction = custom_math.points_clock_direction(curve_end_points)
        self.fill = FILL.ADD if points_direction == 1 else FILL.SUBTRACT
        return

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


    def get_length_worldspace(self):
        return sum(c.get_length_world() for c in self.curves)


    # TODO
    def remove_curve(self, curve):
        return


    # TODO
    def pop_last_curve(self):
        return

    def __len__(self):
        return len(self.curves)

    def get_center(self):
        position_sum = 0
        for curve in self.curves:
            position_sum += np.sum(curve.tween_points, axis=0)
        return position_sum / self.num_points

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

    def get_equally_spaced_points_along(self, num_points,
                                        return_relative_to_upper_left_curve=False):

        resultant_points = []
        curves_and_points = {index: [] for index in range(len(self))}
        # if self.fill == FILL.ADD:
        #     print(len(curves_and_points))

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

    # Drawing
    def draw(self, surface, radius, color=None, width=1):
        # draw curves in colors corresponding to their order in this contour
        no_input_color = color is None
        for i, curve in enumerate(self.curves):
            if no_input_color:
                color = [custom_colors.LT_GRAY,
                                custom_colors.mix_color(custom_colors.GREEN, custom_colors.RED, (i+1)/(len(self)+1)),
                                custom_colors.GRAY]
            curve.draw(surface, radius, color, width=width)

        # draw debug information (center, etc.)
        if globvar.show_extra_curve_information:
            a = 1
            # center = self.get_center()
            # debug_alpha = 0.3
            # s = pygame.Surface((radius * 2, radius * 2))  # the size of your rect
            # s.set_alpha(np.floor(debug_alpha * 255))  # alpha level
            # s.fill(custom_colors.RED)
            # surface.blit(s, center - radius)

            pygame.draw.circle(surface, custom_colors.RED, self.get_upper_left_camera(), radius*0.75)
            pygame.draw.circle(surface, custom_colors.RED, self.get_lower_right_camera(), radius*0.75)
        return

    def draw_control_points(self, surface):
        for curve in self.curves:
            curve.draw_control_points(globvar.screen, radius=globvar.POINT_DRAW_RADIUS)
        return

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


def calc_curve_score_MSE(curve1, curve2):
    if len(curve1.tween_points) != len(curve2.tween_points):
        raise ValueError("Tried to calculate score of curves with different numbers of points")
    # Offset the given points so that they're centered
    c1_tween_points = custom_math.camera_to_worldspace(curve1.tween_points)
    c2_tween_points = custom_math.camera_to_worldspace(curve2.tween_points)
    return -(np.sum((c1_tween_points - c2_tween_points) ** 2) / len(c1_tween_points))


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
For C1 being the contour with at least as many curves as C2:
1) find equally spaced points along each of the curves
2) find the rotational offset that makes the points fit each other best (MSE)
3) return the points for each and the mapping between them; also the MSE score found
"""
def find_pillow_projection_mapping(contour1: Contour, contour2: Contour):
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp

    n1 = len(contour1)
    n2 = len(contour2)

    num_sample_points = globvar.POINTS_TO_GET_CONTOUR_MAPPING_WITH
    contours = [contour1, contour2]
    equidis_point_lists = []
    starting_points = []
    for contour in contours:
        equidis_points = contour.get_equally_spaced_points_along(num_sample_points)

        anchor = contour.get_upper_left_world()         # TODO maybe change this to whichever anchor which minimizes MSE to each contour's center?

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

    score *= -1 / num_sample_points         # lower MSE is better

    if contour1.fill != contour2.fill:
        print("Fills don't match but were the only options! Beware of strange results!")

    return mapping, score


"""
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

        anchor = contour.get_upper_left_world()         # TODO maybe change this to whichever anchor which minimizes MSE to both contours' centers?

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


# Interpolation functions =============================================

"""
Interpolates between two contours using the OferMin method:
use the mapping given to determine which curves between the relevant indices
given in the mapping should be mapped to "zero curves" in c2.
"""

def lerp_contours_pillow_proj(contour1, contour2, point_mapping, t):
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp
        t = 1 - t

    n1 = len(contour1)
    n2 = len(contour2)

    lerped_contour = Contour()

    num_pairs = len(list(point_mapping.keys()))
    lerped_points = []
    for pair in point_mapping.values():
        p1, p2 = pair
        lerped_points.append(custom_math.interpolate_np(p1, p2, t))

    # just make a new contour from quadratic beziers connecting each point linearly
    prev_point = lerped_points[0]
    point = prev_point
    for i in range(num_pairs + 1):
        cyclic_index = i % num_pairs
        prev_point = point
        point = lerped_points[cyclic_index]
        line_points = np.array([point, custom_math.interpolate_np(point, prev_point, 0.5), prev_point])
        new_curve = curve.Bezier(line_points)
        lerped_contour.append_curve(new_curve)

    lerped_contour.update_bounds()
    return lerped_contour


def lerp_contours_relative_proj(contour1, contour2, curve_mapping, t):
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp
        t = 1 - t

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

# TODO fix or remove
# def map_and_lerp(contour1, contour2, lerp_weight, mapping_function, lerping_function):
#     mapping, mapping_score, switched = mapping_function(contour1, contour2)
#     lerped_contour = lerping_function(contour1, contour2, mapping, lerp_weight)
#     return lerped_contour