"""
Represents a Contour object: a path of interconnected Curve objects.
"""
import math

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

        self.em_origin = globvar.empty_offset.copy()

        self.origin_offset = globvar.empty_offset.copy()
        self.scale = 1

        self.fill = FILL.ADD

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

    """
    Move the Contour by some offset
    """
    def set_offset(self, offset_x, offset_y):
        self.origin_offset = np.array([offset_x, offset_y], dtype=globvar.POINT_NP_DTYPE)
        for curve in self.curves:
            curve.set_offset(self.origin_offset)
        return

    # def set_global_offset(self, offset_x, offset_y):
    #     for curve in self.curves:
    #         curve.set_offset(self.origin_offset + np.array([offset_x, offset_y], dtype=globvar.POINT_NP_DTYPE))

    def set_scale(self, scale):
        self.scale = scale
        for curve in self.curves:
            curve.set_scale(self.scale)
        return

    def em_offset(self, offset_x, offset_y):
        offset = np.array([offset_x, offset_y], dtype=globvar.POINT_NP_DTYPE)
        for curve in self.curves:
            curve.em_offset(offset)
        return

    def em_scale(self, scale):
        for curve in self.curves:
            curve.em_scale(scale)

        return

    def append_curve(self, curve):
        # print("appended curve for contour", self, "which before t?he add had", len(self), "curves")
        self.curves.append(curve)
        self.num_points += curve.tween_points.shape[0]
        return

    def append_curve_multi(self, curves):
        for curve in curves:
            self.append_curve(curve)
        return

    def append_curves_from_np(self, curves_point_data):
        for curve_data in curves_point_data:
            self.append_curve(curve.Bezier(curve_data))
        return

    def get_true_points(self):
        string = ""
        for curve in self.curves:
            string += str(np.round(curve.true_points, 3)) + "\n"
        return string

    def get_upper_left(self):
        min_left = math.inf
        min_up = math.inf
        for curve in self.curves:
            up_left = curve.get_upper_left()
            min_left = min(min_left, up_left[0])
            min_up = min(min_up, up_left[1])
        return np.array([min_left, min_up], dtype=globvar.POINT_NP_DTYPE)

    def get_lower_right(self):
        max_right = -math.inf
        max_down = -math.inf
        for curve in self.curves:
            down_right = curve.get_lower_right()
            max_right = max(max_right, down_right[0])
            max_down = max(max_down, down_right[1])
        return np.array([max_right, max_down], dtype=globvar.POINT_NP_DTYPE)

    """
    Check that each curve is connected to the following
    """
    def is_closed(self):
        num_curves = len(self)
        for i in range(num_curves):
            c1 = self.curves[i % num_curves]
            c2 = self.curves[(i+1) % num_curves]
            e1 = c1.points[-1]
            s2 = c2.points[0]
            if not np.all(np.isclose(e1, s2)):
                return False
        return True


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

# Drawing
    def draw(self, surface, radius, color_gradient=True, width=1):
        # draw curves in colors corresponding to their order in this contour
        for i, curve in enumerate(self.curves):
            if color_gradient:
                input_colors = [custom_colors.LT_GRAY,
                                custom_colors.mix_color(custom_colors.GREEN, custom_colors.RED, (i+1)/(len(self)+1)),
                                custom_colors.GRAY]
            curve.draw(surface, radius, input_colors, width=width)

        # draw debug information (center, etc.)
        if globvar.DEBUG:
            center = self.get_center()
            debug_alpha = 0.3
            s = pygame.Surface((radius * 2, radius * 2))  # the size of your rect
            s.set_alpha(np.floor(debug_alpha * 255))  # alpha level
            s.fill(custom_colors.RED)
            surface.blit(s, center - radius)

            pygame.draw.circle(surface, custom_colors.RED, self.get_upper_left(), radius)
            pygame.draw.circle(surface, custom_colors.RED, self.get_lower_right(), radius)
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
    c1_tween_points = (curve1.tween_points / curve1.scale) - curve1.origin_offset
    c2_tween_points = (curve2.tween_points / curve2.scale) - curve2.origin_offset
    return -(np.sum((c1_tween_points - c2_tween_points) ** 2) / len(c1_tween_points))



"""
Let C1 be the contour with more curves and C2 that with less.
Determines the mapping by the OferMin method and returns
the mapping of lines in C2 to those in C1 which are determined to be most relevant
in the form [indices of c1 to take, offset of c2 to use]
"""

def find_ofer_min_mapping(contour1, contour2):
    # make the contour1 variable hold the larger of the two
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp

    # if len(contour2) > len(contour1):
    #     raise AttributeError("Contour 1 needs to have at least as many curves as Contour 2")

    n1 = len(contour1)
    n2 = len(contour2)
    contour1.update_curve_center_relative_angles()
    contour2.update_curve_center_relative_angles()

    best_score = -math.inf
    best_mapping = None

    # To find the best match amongst the curves:
    # 1) pick any n2 of the n1 curves to test a mapping to
    # 2) pick some circular permutation (an offset, basically) of those n2 curves

    # itertools.combinations gives you any subset of r elements from the input iterable
    # crucially, it maintains the order of those elements
    index_subsets = itertools.combinations(range(n1), n2)  # returns all sets of indices in [0, n-1] of size n2
    range_n2 = list(range(n2))
    for indices in index_subsets:
        # iterate through all offsets
        for offset in range_n2:
            # use the offset by iterating through all placements
            # build a score for this mapping (based on both the indices and offset currently being tested)
            current_mapping_score = 0
            for c1_index, c2_index in zip(indices, range_n2):
                c1_curve = contour1.curves[c1_index]
                c2_curve = contour2.curves[(c2_index + offset) % n2]
                current_mapping_score += calc_curve_score_MSE(c1_curve, c2_curve)

            if current_mapping_score > best_score:
                best_score = current_mapping_score
                best_mapping = [indices, offset]
    return best_mapping, best_score


"""
Interpolates between two contours using the OferMin method:
use the mapping given to determine which curves between the relevant indices
given in the mapping should be mapped to "zero curves" in c2.
"""


def lerp_contours_OMin(contour1, contour2, mapping, t, debug_info=False):
    lerped_contour = Contour()

    # we know we need n1 curves in total
    # we also know which indices of c1 were chosen
    # therefore for any index not after the expected one (i.e. there is a gap between one index and the next)
    # we fill with a "zero curve"

    c1_chosen_curves, c2_offset = mapping

    # make the same switch as the mapping did; from here, everything should be the same
    if len(contour2) > len(contour1):
        temp = contour1
        contour1 = contour2
        contour2 = temp
        t = 1-t

    n1 = len(contour1)
    n2 = len(contour2)

    current_c2_index = 0
    last_endpoint = contour2.curves[current_c2_index + c2_offset].true_points[0]
    if debug_info:
        print("making", lerped_contour, "...")
    for expected_index in range(n1):
        if expected_index in c1_chosen_curves:
            if debug_info:
                print("adding curve #", expected_index, ", a mapping between C2's", current_c2_index, "and C1's",
                      expected_index)
            # interestingly, putting this line here adds some weird rotations to the transformation
            # current_c2_index += 1
            circular_c2_index = (current_c2_index + c2_offset) % n2
            # now add the curve you've been trying to give me (we're all caught up to the expected index)
            # by interpolating between the current c2 curve and the corresponding c1 curve
            lerped_points = custom_math.interpolate_np(contour1.curves[expected_index].true_points,
                                           contour2.curves[circular_c2_index].true_points,
                                           t)
            last_endpoint = contour2.curves[circular_c2_index].true_points[-1]
            lerped_contour.append_curve(curve.Bezier(lerped_points))
            current_c2_index += 1
        else:
            if debug_info:
                print("adding curve #", expected_index, ", a mapping between a null curve and C1's", expected_index)
            # interpolate between the current c1_curve and the "zero curve" which
            # consists of four points bunched together at c2's last point
            lerped_zero_curve_points = custom_math.interpolate_np(
                contour1.curves[expected_index].true_points,
                np.array([last_endpoint, last_endpoint, last_endpoint, last_endpoint]),
                t)
            lerped_contour.append_curve(curve.Bezier(lerped_zero_curve_points))

    return lerped_contour

# TODO fix or remove
# def map_and_lerp(contour1, contour2, lerp_weight, mapping_function, lerping_function):
#     mapping, mapping_score, switched = mapping_function(contour1, contour2)
#     lerped_contour = lerping_function(contour1, contour2, mapping, lerp_weight)
#     return lerped_contour