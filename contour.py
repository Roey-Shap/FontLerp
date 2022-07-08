"""
Represents a Contour object: a path of interconnected Curve objects.
"""

import pygame

import bezier
import global_variables as globvar
import numpy as np
import itertools
import custom_colors

class Contour(object):
    def __init__(self):
        self.curves = []
        self.num_points = 0
        self.position_sum = np.array([0, 0], dtype=globvar.POINT_NP_DTYPE)
        globvar.contours.append(self)
        return

    """
    Move the Contour by some offset
    """
    def offset(self, offset_x, offset_y):
        for curve in self.curves:
            curve.offset(offset_x, offset_y)
        return

    def append_curve(self, curve):
        print("appended curve for contour", self, "which before the add had", len(self), "curves")
        self.curves.append(curve)
        self.num_points += curve.points.shape[0]
        return

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
            position_sum += np.sum(curve.points, axis=0)
        return position_sum / self.num_points

    def get_curve_center_relative_angles(self):
        center = self.get_center()
        for curve in self.curves:
            # offset the curve's center to "(0, 0)" by subtracting 'center'
            # note that we work in degrees
            adjusted_point = curve.average_point - center
            cur_curve_rel_angle = np.arctan(adjusted_point[1] / adjusted_point[0]) * 180 / np.pi
            curve.current_contour_relative_angle = cur_curve_rel_angle
            # print("rel angle:", cur_curve_rel_angle)

# Drawing
    def draw(self, surface, radius, color_gradient=True):
        for i, curve in enumerate(self.curves):
            if color_gradient:
                input_colors = [custom_colors.LT_GRAY,
                                custom_colors.mix_color(custom_colors.BLUE, custom_colors.WHITE, i/len(self)),
                                custom_colors.GRAY]
            curve.draw(surface, radius, input_colors)

        if globvar.DEBUG:
            center = self.get_center()
            debug_alpha = 0.3
            s = pygame.Surface((radius * 2, radius * 2))  # the size of your rect
            s.set_alpha(np.floor(debug_alpha * 255))  # alpha level
            s.fill(custom_colors.RED)
            surface.blit(s, center - radius)
        return


def interpolate_np(ndarr1, ndarr2, t):
    weighted_1 = ndarr1 * t
    weighted_2 = ndarr2 * (1-t)
    return weighted_1 + weighted_2


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
Let C1 be the contour with more curves and C2 that with less.
Determines the mapping by the OferMin method and returns
the mapping of lines in C2 to those in C1 which are determined to be most relevant
in the form [indices of c1 to take, offset of c2 to use]
"""


def ofer_min(contour1, contour2):
    # make the contour1 variable hold the larger of the two
    if len(contour2) > len(contour1):
        raise AttributeError("Contour 1 needs to have at least as many curves as Contour 2")
        # temp = contour1
        # contour1 = contour2
        # contour2 = temp
    n1 = len(contour1)
    n2 = len(contour2)
    contour1.get_curve_center_relative_angles()
    contour2.get_curve_center_relative_angles()

    best_score = 0
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
                current_mapping_score += calc_score(c1_curve, c2_curve)

            if current_mapping_score > best_score:
                best_score = current_mapping_score
                best_mapping = [indices, offset]

    return best_mapping


"""
Interpolates between two contours using the OferMin method:
use the mapping given to determine which curves between the relevant indices
given in the mapping should be mapped to "zero curves" in c2.
"""


def lerp_contours_OMin(contour1, contour2, mapping, t):
    lerped_contour = Contour()
    n1 = len(contour1)
    n2 = len(contour2)

    # we know we need n1 curves in total
    # we also know which indices of c1 were chosen
    # therefore for any index not after the expected one (i.e. there is a gap between one index and the next)
    # we fill with a "zero curve"

    c1_chosen_curves, c2_offset = mapping
    print(mapping)
    expected_index = 0
    last_endpoint = contour2.curves[0 + c2_offset].points[0]
    # note that current_c2_index is just what number curve we're on in the c1_chosen_curves array
    for current_c2_index, c1_index in enumerate(c1_chosen_curves):
        while expected_index < c1_index:
            # interpolate between the current c1_curve and the "zero curve" which
            # consists of four points bunched together at c2's last point
            lerped_zero_curve_points = interpolate_np(contour1.curves[c1_index].points,
                                           np.array([last_endpoint, last_endpoint, last_endpoint, last_endpoint]),
                                           t)
            lerped_contour.append_curve(bezier.Bezier(lerped_zero_curve_points))
            expected_index += 1

        circular_c2_index = (current_c2_index + c2_offset) % n2
        # now add the index you're trying to give me (we're all caught up to the expected index)
        # by interpolating between the current c2 curve and the corresponding c1 curve
        lerped_points = interpolate_np(contour1.curves[c1_index].points,
                                       contour2.curves[circular_c2_index].points,
                                       t)
        last_endpoint = contour2.curves[circular_c2_index].points[-1]
        lerped_contour.append_curve(bezier.Bezier(lerped_points))

    return lerped_contour

