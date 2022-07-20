import numpy as np
import pygame
import custom_colors
import global_variables as globvar
import point as AbsPoint
import copy

""" 
Represents a 2D Cubic Bezier curve
Is comprised of 4 control points: p0, p1, p2, p3
for a percentage t in [0, 1], we have a Bernstein Polynomial:
P(t) = 
p0(-t^3 + 3t^2 - 3t + 1) + 
p1(3t^3 - 6t^2 + 3t) +
p2(-3t^3 + 3t^2) +
p3(t^3)

= t^3(-p0 + 3p1 - 3p2 + p3) +
  t^2(3p0 - 6p1 + 3p2) +
  t(-3p0 + 3p1) + 
  1

"""


class Curve(object):
    def __init__(self, points):
        if not isinstance(points, np.ndarray):
            raise ValueError("tried inputting: \'", points, "\' as an ndarray when it wasn't one")

        self.num_points = points.shape[0]

        self.true_points = points.copy()
        self.points = points.astype(globvar.POINT_NP_DTYPE)
        self.origin_offset = np.array([0, 0], dtype=globvar.POINT_NP_DTYPE)
        self.scale = 1

        self.abstract_points = []
        self.average_point = None

        self.current_contour_relative_angle = None

        for i, point in enumerate(self.points):
            abs_point = AbsPoint.Point(point, (i == 0 or i == self.num_points-1))
            # add this abstract point to this Bezier's list of points
            self.abstract_points.append(abs_point)

            # mark this abstract point's existence global
            globvar.abstract_points.append(abs_point)

        globvar.curves.append(self)
        return

    def destroy(self):
        for point in self.abstract_points:
            point.destroy()
        index = globvar.curves.index(self)
        globvar.curves.pop(index)
        return

    def copy(self):
        clone = copy.deepcopy(self)
        globvar.curves.append(clone)
        return clone

    def set_offset(self, origin_offset):
        # offset the current points (4x2)
        self.origin_offset = origin_offset
        # self.offset_points = (self.points * self.scale) + self.origin_offset
        # use the offset points to update all relevant data
        self.update_points(self.true_points)
        return

    def set_scale(self, scale):
        self.scale = scale
        self.update_points(self.true_points)
        return

    # offset true points around (so different curves and contours can have different offsets from their parent glyph)
    def em_offset(self, offset):
        self.true_points += offset
        self.update_points(self.true_points)
        return

    # scale true points
    def em_scale(self, scale):
        self.true_points *= scale
        self.update_points(self.true_points)
        return


    def get_dimensions(self):
        return np.max(self.points, axis=0) - np.min(self.points, axis=0)

    def get_upper_left(self):
        return np.min(self.points, axis=0)

    def get_lower_right(self):
        return np.max(self.points, axis=0)


    """
    Return the angle of line formed by the endpoints in degrees
    """
    def get_angle(self):
        p1 = self.points[0]
        p2 = self.points[-1]
        difference_vector = p1-p2
        return np.arctan(difference_vector[1] / difference_vector[0]) * 180 / np.pi

# Point Manipulation
    """
    For each point, check if the mouse is hovering over it
    """
    def check_abstract_points(self, r):
        change_detected = False
        for i, abs_point in enumerate(self.abstract_points):
            current_change = abs_point.changed_position_this_frame
            change_detected = change_detected or current_change
            if current_change:
                self.true_points[i] = (abs_point.np_coords()/self.scale) - self.origin_offset

        if change_detected:
            # take the self.points that were updated and recalculate the bernstein points
            self.update_points(self.true_points)

        return

    """
    Draw the "control points" which influence the Bezier's shape in between the endpoints
    """
    def draw_control_points(self, surface, color, radius=1):
        for abs_point in self.abstract_points:
            abs_point.draw(surface, radius)
        return

    def step(self):
        return


# class Line(Curve):
#     def __init__(self, points):
#         super().__init__(points)
#
#         if points.shape[0] != 2:
#             raise ValueError("Linear Curves must have 2 points: matrix", points,
#                              "was input with:", points.shape[0], "points")
#
#         self.update_points(points)
#
#     def update_points(self, points):
#         self.points = points
#         for i, abstract_point in enumerate(self.abstract_points):
#             abstract_point.update_coords(self.points[i])
#         self.average_point = np.average(self.points, axis=0)
#         return
#
#     def get_length(self):
#         return np.linalg.norm(self.points[0]-self.points[1])
#
#     def clone(self):
#         return Line(np.ndarray.copy(self.points))
#
# # Drawing
#     def draw(self, surface, point_radius):
#         self.draw_tween_lines(surface, custom_colors.BLACK)
#         self.draw_control_points(surface, custom_colors.LT_GRAY, radius=point_radius)
#
#     def draw_tween_lines(self, surface, color):
#         pygame.draw.line(surface, color, self.points[0], self.points[1])
#


class Bezier(Curve):
    adjustment_matrix_cubic = np.array([[-1, 3, -3, 1],
                                        [3, -6, 3, 0],
                                        [-3, 3, 0, 0],
                                        [1, 0, 0, 0]])

    adjustment_matrix_quadratic = np.array([[1, -2, 1],
                                            [-2, 2, 0],
                                            [1, 0, 0]])

    adjustment_matrices = [adjustment_matrix_quadratic, adjustment_matrix_cubic]
    """
    Constructs a Bezier object from a set of points matrix,
    where each row holds a point. 
    
    This is converted so that each row holds the values of that dimension (for matrix multiplication ease later, 
    it's a bit easier on the eye in ML-fashion: x, y as feature columns, but whatever)
    e.g:
    x1, x2, x3, ..., xm
    y1, y2, y3, ..., ym
    """
    def __init__(self, points):
        super().__init__(points)

        if points.shape[0] < 3 or points.shape[0] > 4:
            raise ValueError("Beziers must have 3 or 4 points: matrix", points,
                             "was input with:", points.shape[0], "points")

        calc_functions = [self.calc_point_quadratic, self.calc_point_cubic]

        self.adjustment_matrix = Bezier.adjustment_matrices[self.num_points-3]
        self.calc_point_function = calc_functions[self.num_points-3]

        self.bernstein_points = None
        self.tween_points = None

        self.update_points(points)
        return

    """
    Sets this Bezier's points and caches linear combinations of them for interpolated points.
    Returns a (2x4) matrix
    ===
    the adjusted Berstein polynomial form above yields 
    an inner product of (a vector of powers of t) and (linear combinations of the control points)
    cache those constant linear combinations of the control points for calculating points in between 
    """
    def update_points(self, true_points):
        self.true_points = true_points
        self.points = (self.true_points + self.origin_offset) * self.scale
        for i, abstract_point in enumerate(self.abstract_points):
            abstract_point.update_coords(self.points[i])

        # turn pretty (nx2) points into (2xn) via transpose
        # (2xn) = (2xn)*(nxn)
        self.bernstein_points = np.matmul(self.points.T, self.adjustment_matrix)
        self.calc_tween_points()
        self.average_point = np.average(self.points, axis=0)
        return

    """
    Calculates a 2D point resulting from the weighted interpolation of the control points.
    Returns a (2x1) matrix (a single point)
    """
    def calc_point(self, t):
        return self.calc_point_function(t)

    def calc_point_quadratic(self, t):
        ts = np.array([t*t, t, 1])
        # return (2x1) = (2x3)*(3x1)
        return np.matmul(self.bernstein_points, ts)

    def calc_point_cubic(self, t):
        t_squared = t * t
        t_cubed = t_squared * t
        ts = np.array([t_cubed, t_squared, t, 1], dtype=globvar.POINT_NP_DTYPE)
        # return (2x1) = (2x4)*(4x1)
        return np.matmul(self.bernstein_points, ts)

    def calc_tween_points(self):
        size_adj = 0 if self.num_points == 4 else 1
        self.tween_points = np.matmul(self.bernstein_points, globvar.t_values[:, size_adj:4].T).T
        return
        # self.render_points = np.zeros((accuracy, 2))
        # self.render_points = []
        # # TODO cap accuracy based on distance of points? Less are needed in some situations
        # for i in range(accuracy+1):
        #     t = i/accuracy
        #     p = self.calc_point(t)
        #     self.render_points.append(p)

    def get_length(self):
        sigma = 0
        num_render_points = len(self.tween_points)
        for i in range(num_render_points-1):
            p1 = self.tween_points[i]
            p2 = self.tween_points[i+1]
            sigma += np.linalg.norm(p1-p2)

        return sigma

    def step(self):

        return

# Drawing
    def draw(self, surface, point_radius, input_colors=None, width=1):
        colors = [custom_colors.LT_GRAY, custom_colors.BLACK, custom_colors.BLUE]
        if input_colors is not None:
            colors = input_colors
        self.draw_tween_lines(surface, colors[1], width=width)
        if globvar.DEBUG:
            self.draw_control_lines(surface, colors[0], width=width)
            self.draw_control_points(surface, colors[2], radius=point_radius)
    """
    Draws the lines connected the precomputed "render points
    """
    def draw_tween_lines(self, surface, color, width=1):
        num_render_points = len(self.tween_points)
        p1 = self.tween_points[0]
        p2 = p1
        for i in range(num_render_points-1):
            p1 = p2
            p2 = self.tween_points[i+1]
            pygame.draw.line(surface, color, p1, p2, width=width)

        return

    """
    Draw the lines from the control points to show how they influence the Bezier's curvature
    """
    def draw_control_lines(self, surface, color, width=1):
        width = round(width/3)
        pygame.draw.line(surface, color, self.points[0], self.points[1], width=width)
        pygame.draw.line(surface, color, self.points[-2], self.points[-1], width=width)
        return

    """
    Draws the precomputed "render points" which approximate the Bezier 
    """
    def draw_render_points(self, surface, color, radius=1):
        for point in self.tween_points:
            p = pygame.math.Vector2(point[0], point[1])
            pygame.draw.circle(surface, color, p, radius)

        return



