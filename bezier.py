import numpy as np
import pygame
import custom_colors
import global_variables as globvar
import point as AbsPoint

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
        self.points = points.astype(globvar.POINT_NP_DTYPE)
        self.abstract_points = []
        for point in points:
            abs_point = AbsPoint.Point(point)
            # add this abstract point to this Bezier's list of points
            self.abstract_points.append(abs_point)
            # mark this abstract point's existence global
            globvar.abstract_points.append(abs_point)

    def get_dimensions(self):
        return np.max(self.points, axis=0) - np.min(self.points, axis=0)

    def get_upper_left(self):
        return np.min(self.points, axis=0)

    def get_lower_right(self):
        return np.max(self.points, axis=0)

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
                self.points[i] = abs_point.np_coords()

        if change_detected:
            # take the self.points that were updated and recalculate the bernstein points
            self.update_points(self.points)

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


class Line(Curve):
    def __init__(self, points):
        super().__init__(points)

        if points.shape[0] != 2:
            raise ValueError("Linear Curves must have 2 points: matrix", points,
                             "was input with:", points.shape[0], "points")

    def update_points(self, points):
        self.points = points
        for i, abstract_point in enumerate(self.abstract_points):
            abstract_point.update_coords(self.points[i])
        return

    def offset(self, offset_x=0, offset_y=0):
        # offset the current points (4x2)
        offset_array = np.array([offset_x, offset_y])

        # use the offset points to update all relevant data
        self.update_points(self.points + offset_array)
        return

    def copy(self):
        return Line(np.ndarray.copy(self.points))

# Drawing
    def draw(self, surface, point_radius):
        self.draw_tween_lines(surface, custom_colors.BLACK)
        self.draw_control_points(surface, custom_colors.LT_GRAY, radius=point_radius)

    def draw_tween_lines(self, surface, color):
        pygame.draw.line(surface, color, self.points[0], self.points[1])





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

        control_line_functions = [self.draw_control_lines_quadratic, self.draw_control_lines_cubic]
        calc_functions = [self.calc_point_quadratic, self.calc_point_cubic]

        self.adjustment_matrix = Bezier.adjustment_matrices[self.num_points-3]
        self.calc_point_function = calc_functions[self.num_points-3]
        self.draw_control_lines_function = control_line_functions[self.num_points-3]

        self.bernstein_points = None
        self.render_points = None

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
    def update_points(self, points):
        self.points = points
        for i, abstract_point in enumerate(self.abstract_points):
            abstract_point.update_coords(self.points[i])
        # turn pretty (nx2) points into (2xn) via transpose
        # (2xn) = (2xn)*(nxn)
        self.bernstein_points = np.matmul(self.points.T, self.adjustment_matrix)
        return

    def offset(self, offset_x=0, offset_y=0):
        # offset the current points (4x2)
        offset_array = np.array([offset_x, offset_y])

        # use the offset points to update all relevant data
        self.update_points(self.points + offset_array)
        return

    def copy(self):
        return Bezier(np.ndarray.copy(self.points))

    """
    Calculates a 2D point resulting from the weighted interpolation of the control points.
    Returns a (2x1) matrix (a single point)
    """
    # TODO vectorize to allow for an arbitrary number of points and not just 1 (?)
    def calc_point(self, t):
        return self.calc_point_function(t)

    def calc_point_quadratic(self, t):
        ts = np.array([t*t, t, 1])
        # return (2x1) = (2x4)*(4x1)
        return np.matmul(self.bernstein_points, ts)

    def calc_point_cubic(self, t):
        t_squared = t * t
        t_cubed = t_squared * t
        ts = np.array([t_cubed, t_squared, t, 1])
        # return (2x1) = (2x4)*(4x1)
        return np.matmul(self.bernstein_points, ts)

    def calc_tween_points(self, accuracy):
        # self.render_points = np.zeros((accuracy, 2))
        self.render_points = []
        # TODO cap accuracy based on distance of points? Less are needed in some situations
        for i in range(accuracy+1):
            t = i/accuracy
            p = self.calc_point(t)
            self.render_points.append(pygame.math.Vector2(p[0], p[1]))

        return

    def step(self):
        self.calc_tween_points(globvar.bezier_accuracy)

        return

# Drawing
    def draw(self, surface, point_radius):
        self.draw_control_lines(surface, custom_colors.LT_GRAY)
        self.draw_tween_lines(surface, custom_colors.BLACK)
        self.draw_control_points(surface, custom_colors.LT_GRAY, radius=point_radius)
    """
    Draws the lines connected the precomputed "render points
    """
    def draw_tween_lines(self, surface, color, width=1):
        num_render_points = len(self.render_points)
        p1 = self.render_points[0]
        p2 = p1
        for i in range(num_render_points-1):
            p1 = p2
            p2 = self.render_points[i+1]
            pygame.draw.line(surface, color, p1, p2)

        return

    """
    Draw the lines from the control points to show how they influence the Bezier's curvature
    """
    def draw_control_lines(self, surface, color):
        self.draw_control_lines_function(surface, color)

    def draw_control_lines_quadratic(self, surface, color):
        pygame.draw.line(surface, color, self.points[0], self.points[1])
        pygame.draw.line(surface, color, self.points[1], self.points[2])
        return

    def draw_control_lines_cubic(self, surface, color):
        pygame.draw.line(surface, color, self.points[0], self.points[1])
        pygame.draw.line(surface, color, self.points[2], self.points[3])
        return

    """
    Draws the precomputed "render points" which approximate the Bezier 
    """
    def draw_render_points(self, surface, color, radius=1):
        for point in self.render_points:
            p = pygame.math.Vector2(point[0], point[1])
            pygame.draw.circle(surface, color, p, radius)



