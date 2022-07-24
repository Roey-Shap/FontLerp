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
        anchor = self.get_lower_right_world()
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

    def draw_filled_polygon(self, surface, fill_color, width=1):
        # get all of the points of this contour's lines
        all_tween_points = []
        for curve in self.curves:
            for tween_point in curve.tween_points:
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


"""
Let C1 be the contour with more curves and C2 that with less.
Determines the mapping by the OferMin method and returns
the mapping of lines in C2 to those in C1 which are determined to be most relevant
in the form [indices of c1 to take, offset of c2 to use]
"""

def find_ofer_min_mapping(contour1: Contour, contour2: Contour):
    if contour1.fill != contour2.fill:
        return None, -math.inf


    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp

    n1 = len(contour1)
    n2 = len(contour2)
    contours = [contour1, contour2]

    # find the mapping by matching points from n2 into n1
    # given a curve in n2, pick the curve in c1 who the majority of its points fall into

    equidistant_points_dicts = []
    # now we need to know what overall t values and for what curves we need to get each point to be
    # the same length along the contour
    for contour_index in [0, 1]:
        contour = contours[contour_index]
        num_curves = len(contour)

        num_points = globvar.POINTS_TO_GET_CONTOUR_MAPPING_WITH
        adjusted_points_in_curve_groups = contour.get_equally_spaced_points_along(num_points,
                                                                                  return_relative_to_upper_left_curve=True)
        # we now have the points; order them such that their indices are from top left and around
        # upper left curve should be 0 and its points 0, 1, 2, etc.

        equidistant_points_dicts.append(adjusted_points_in_curve_groups)

        # # we'll later need the curve containing the upper-left-most point for synchronization
        # temp_points_dict = custom_math.sort_dictionary({point: np.linalg.norm(point) for point in equally_spaced_points_dict},
        #                                                by_key_or_value=1)
        # upper_left_point = next(temp_points_dict.keys())
        # upper_left_points.append(next(temp_points_dict.keys()))     # gets the first element of this list - the smallest-norm point



    # for each of the curves in C2 which of those in
    # C1 contains the largest number of the same point indices as it does

    mapping_dict = {c1: None for c1 in range(n1)}
    curves_mapped = 0
    points_correctly_mapped = 0

    c1_dict, c2_dict = equidistant_points_dicts
    for c2_index in range(n2):
        c2_points_in_curve = c2_dict[c2_index]
        scores_against_c1_curves = {c1: 0 for c1 in range(n1)}

        best_C1_index = 0

        for c1_index in range(n1):
            c1_points_in_curve = [adj_index for adj_index, coords in c1_dict[c1_index]]
            current_number_of_points_matched = 0

            for c2_point_index, c2_point_coords in c2_points_in_curve:
                if c2_point_index in c1_points_in_curve:
                    current_number_of_points_matched += 1

            scores_against_c1_curves[c1_index] = current_number_of_points_matched
            #
            # if current_number_of_points_matched > highest_number_of_points_matched:
            #     best_C1_index = c1_index
            #     highest_number_of_points_matched = current_number_of_points_matched

        # pick the best available C1 curve
        scores_against_c1_curves = custom_math.sort_dictionary(scores_against_c1_curves, by_key_or_value=1, reverse=True)
        for c1_index, c1_score in enumerate(scores_against_c1_curves):
            if mapping_dict[c1_index] is None:
                best_C1_index = c1_index
                curves_mapped += 1
                points_correctly_mapped += c1_score
                break

        mapping_dict[best_C1_index] = c2_index
    print("Mapped all curves: ", curves_mapped == n2)

    return mapping_dict, points_correctly_mapped


def find_ofer_min_mapping_old(contour1, contour2):
    # make the contour1 variable hold the larger of the two
    if len(contour1) > globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD:
        print("WARNING: Contour1 had more than", globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD, "curves")
    if len(contour2) > globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD:
        print("WARNING: Contour2 had more than", globvar.CONTOUR_CURVE_AMOUNT_THRESHOLD, "curves")

    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp

    n1 = len(contour1)
    n2 = len(contour2)


    # contour1.update_curve_center_relative_angles()
    # contour2.update_curve_center_relative_angles()

    best_score = -math.inf
    best_mapping = None

    # To find the best match amongst the curves:
    # 1) pick any n2 of the n1 curves to test a mapping to
    # 2) pick some circular permutation (an offset, basically) of those n2 curves

    # itertools.combinations gives you any subset of r elements from the input iterable
    # crucially, it maintains the order of those elements
    index_subsets = itertools.combinations(range(n1), n2)  # returns all sets of indices in [0, n-1] of size n2
    range_n2 = list(range(n2))
    outer_iterations = custom_math.ncr(n1, n2)
    print("Number of index subsets is:", outer_iterations)
    did_it = False
    maximal_tolerable_iterations = 25000

    if outer_iterations > maximal_tolerable_iterations:
        print("Warning: number of possible mappings is too high to find optimal mapping; using greedy method")

        # calculate how many curves we can afford to take on
        sufficient_curves = maximal_curves_for_threshold(n1, n2, maximal_tolerable_iterations)

        # calculate scores
        scores = {}
        average_c1_scores = {}
        best_c1_scores = {c1: -math.inf for c1 in range(n1)}
        for c1_index in range(n1):
            average_score = 0
            c1_curve = contour1.curves[c1_index]
            for c2_index in range_n2:
                c2_curve = contour2.curves[c2_index]
                score = calc_curve_score_MSE(c1_curve, c2_curve)
                average_score += score
                scores[(c1_index, c2_index)] = score
                best_c1_scores[c1_index] = max(best_c1_scores[c1_index], score)

            average_score /= n2
            average_c1_scores[c1_index] = average_score

        sorted_c1_average_scores = custom_math.sort_dictionary(average_c1_scores, 1, reverse=True)

        best_c1_indices = []
        for i, key in enumerate(sorted_c1_average_scores):
            if i < sufficient_curves:
                best_c1_indices.append(key)

        # best_c1_indices = []
        # sorted_c1_scores = custom_math.sort_dictionary(best_c1_scores, 1, reverse=True)
        # for i, key in enumerate(sorted_c1_scores):
        #     if i < sufficient_curves:
        #         best_c1_indices.append(key)

        index_subsets = itertools.combinations(best_c1_indices, n2)
        did_it = True
        outer_iterations = custom_math.ncr(sufficient_curves, n2)
        #
        # scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        # print(scores)
        # # different method interrupting here: find the highest-scoring mapping for each of C2's curves
        # # taken_c1_curves = []
        # # cur_score = 0
        # # for c2_index in range_n2:
        # #     for i, curve_pair in enumerate(scores.keys()):
        # #         if curve_pair[1] == c2_index and curve_pair[0] not in taken_c1_curves:
        # #             taken_c1_curves.append(curve_pair[0])
        # #             cur_score += scores[curve_pair]
        # # return [taken_c1_curves, range_n2.copy()], cur_score
        #
        # best_score = -math.inf
        # best_mapping = []
        # # begin by picking a C2 curve to give "priority to" (be the starting point)
        # for c2_index in range_n2:
        #     # try to build a list of the best-scored pairs that follow cyclic order of the curves in their contours
        #     # find the best-scored pair using c2_index
        #     tolerance = 1000                              # describes how many places down you're willing to go
        #     tolerance_step = 2                           # to find a starting point for a given starting c2_index
        #     for t in range(0, tolerance, tolerance_step):
        #         tolerance_countdown = 0
        #
        #         current_total_score = 0
        #         current_mapping_indices = []
        #         current_mapping_offset = c2_index
        #
        #         current_pair_index = 0
        #         first_c1_chosen = -1
        #         first_c2_chosen = c2_index
        #         last_c1_chosen = -1
        #         last_c2_chosen = c2_index
        #         yet_wrapped = False
        #
        #         for i, curve_pair in enumerate(scores.keys()):
        #             if curve_pair[1] == c2_index:
        #                 tolerance_countdown += 1
        #                 if tolerance_countdown >= t:
        #                     first_c1_chosen = curve_pair[0]
        #                     first_c2_chosen = curve_pair[1]
        #                     last_c1_chosen = curve_pair[0]
        #                     last_c2_chosen = curve_pair[1]
        #                     current_pair_index = i
        #                     current_total_score += scores[curve_pair]
        #                     current_mapping_indices.append(last_c1_chosen)
        #                     break
        #
        #         # go over the remaining curves and find the best one that complies with the order
        #         curve_pairs_list = list(scores.keys())
        #         curves_mapped = 1                                                           # just the initial c2_curve was found
        #         iterations = 0
        #         while curves_mapped < n2 and iterations < len(curve_pairs_list):
        #             iterations += 1
        #             current_pair_index = (current_pair_index + 1) % len(curve_pairs_list)   # we start at current_pair_index so immediately increment
        #             curve_pair = curve_pairs_list[current_pair_index]
        #             cur_c1_curve, cur_c2_curve = curve_pair
        #
        #             curve_is_earlier_in_contour = cur_c1_curve < first_c1_chosen and not yet_wrapped
        #             curve_is_later_in_contour = cur_c1_curve > last_c1_chosen
        #             wrapped_and_is_later = yet_wrapped and curve_is_later_in_contour and cur_c1_curve < first_c1_chosen
        #             no_wrap_and_is_later = (not yet_wrapped) and curve_is_later_in_contour
        #             valid_c1_index = curve_is_earlier_in_contour or wrapped_and_is_later or no_wrap_and_is_later
        #             valid_c2_index = cur_c2_curve == ((last_c2_chosen + 1) % n2)
        #
        #             if valid_c1_index and valid_c2_index:
        #                 if cur_c1_curve < first_c1_chosen:
        #                     yet_wrapped = True
        #
        #                 last_c1_chosen = cur_c1_curve
        #                 last_c2_chosen = cur_c2_curve
        #                 curves_mapped += 1
        #                 current_total_score += scores[curve_pair]
        #                 current_mapping_indices.append(last_c1_chosen)
        #
        #         print(c2_index, current_mapping_indices, "with", curves_mapped, "curves mapped")
        #         # we've found this potential mapping; is its total score high enough?
        #         if current_total_score > best_score and curves_mapped == n2:
        #             best_score = current_total_score
        #             current_mapping_indices.sort()
        #             best_mapping = [current_mapping_indices, current_mapping_offset]
        #
        # # print("Unoptimal OMin mapping found!")
        # return best_mapping, best_score

    # END UNOPTIMAL CASE ====================================
        # random_indices = random.sample(list(range(n1)), n2)
        # random_offset = range_n2[random.randint(0, n2-1)]
        # current_mapping_score = 0
        #
        #
        # for c1_index, c2_index in zip(random_indices, range_n2):
        #     c1_curve = contour1.curves[c1_index]
        #     c2_curve = contour2.curves[(c2_index + random_offset) % n2]
        #     current_mapping_score += calc_curve_score_MSE(c1_curve, c2_curve)
        # return [random_indices, random_offset], current_mapping_score

    # START OPTIMAL CASE ====================================
    print("NOT HERE!!!")
    if did_it:
        print("Completed filtering to less curves: New Outer-iters:", outer_iterations)

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
Interpolates between two contours using the OferMin method:
use the mapping given to determine which curves between the relevant indices
given in the mapping should be mapped to "zero curves" in c2.
"""


def lerp_contours_OMin(contour1, contour2, mapping: dict, t, debug_info=False):
    # we know we need n1 curves in total
    # we also know which indices of c1 were chosen
    # therefore for any index not after the expected one (i.e. there is a gap between one index and the next)
    # we fill with a "zero curve"

    # mapping is a dictionary where each entry is of the form (c1, c2) and if that c1 curve has no mapping, (c1, None)

    # make the same switch as the mapping did; from here, everything should be the same
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp
        t = 1-t

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



# TODO fix or remove
# def map_and_lerp(contour1, contour2, lerp_weight, mapping_function, lerping_function):
#     mapping, mapping_score, switched = mapping_function(contour1, contour2)
#     lerped_contour = lerping_function(contour1, contour2, mapping, lerp_weight)
#     return lerped_contour