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


class BezierCubic2D(object):
    # Note to self: instead of using this's T in bernstein points, without that
    # it seems to form a loop. Interesting...
    adjustment_matrix = np.array([[-1, 3, -3, 1],
                                  [3, -6, 3, 0],
                                  [-3, 3, 0, 0],
                                  [1, 0, 0, 0]])

    """
    Constructs a Cubic Bezier object from a set of points matrix,
    where each row holds a point. 
    
    This is converted so that each row holds the values of that dimension (for matrix multiplication ease later, 
    it's a bit easier on the eye in ML-fashion: x, y as feature columns, but whatever)
    e.g:
    x1, x2, x3, ..., xm
    y1, y2, y3, ..., ym
    """
    def __init__(self, points):
        if not isinstance(points, np.ndarray):
            raise ValueError("tried inputting: \'", points, "\' as a ndarray when it wasn't one")

        if points.shape[0] != 4:
            raise ValueError("Cubic Bezier must have exactly 4 points: vector", points, "was input")

        points = points.astype(globvar.POINT_NP_DTYPE)
        self.points = None
        self.abstract_points = []
        for point in points:
            abs_point = AbsPoint.Point(point)
            # add this abstract point to this Bezier's list of points
            self.abstract_points.append(abs_point)
            # mark this abstract point's existence global
            globvar.abstract_points.append(abs_point)

        self.bernstein_points = None
        self.render_points = None

        self.update_bernstein_points(points)
        return

    """
    Sets this Bezier's points and caches linear combinations of them for interpolated points.
    Returns a (2x4) matrix
    ===
    the adjusted Berstein polynomial form above yields 
    an inner product of (a vector of powers of t) and (linear combinations of the control points)
    cache those constant linear combinations of the control points for calculating points in between 
    """
    def update_bernstein_points(self, points):
        self.points = points
        for i, abstract_point in enumerate(self.abstract_points):
            abstract_point.update_coords(self.points[i])
        # turn pretty (4x2) points into (2x4) via transpose
        # (2x4) = (2x4)*(4x4)
        self.bernstein_points = np.matmul(self.points.T, BezierCubic2D.adjustment_matrix)
        return

    def get_dimensions(self):
        return np.max(self.points, axis=0) - np.min(self.points, axis=0)

    def get_upper_left(self):
        return np.min(self.points, axis=0)

    def get_lower_right(self):
        return np.max(self.points, axis=0)

    def offset(self, offset_x=0, offset_y=0):
        # offset the current points (4x2)
        offset_array = np.array([offset_x, offset_y])

        # use the offset points to update all relevant data
        self.update_bernstein_points(self.points + offset_array)
        return

    def copy(self):
        return BezierCubic2D(np.ndarray.copy(self.points))

    """
    Calculates a 2D point resulting from the weighted interpolation of the control points.
    Returns a (2x1) matrix (a single point)
    """
    # TODO vectorize to allow for an arbitrary number of points and not just 1
    def calc_point(self, t):
        t_squared = t * t
        t_cubed = t_squared * t
        ts = np.array([t_cubed, t_squared, t, 1])
        # return (2x1) = (2x4)*(4x1)
        return np.ndarray.flatten(np.matmul(self.bernstein_points, ts))

    def calc_point_slow(self, t):
        t_squared = t*t
        t_cubed = t_squared*t
        sigma = [0, 0]
        for i in [0, 1]:
            sigma[i] += self.points[0][i] * (-t_cubed + 3*t_squared - 3*t + 1)
            sigma[i] += self.points[1][i] * (3*t_cubed - 6*t_squared + 3*t)
            sigma[i] += self.points[2][i] * (-3*t_cubed + 3*t_squared)
            sigma[i] += self.points[3][i] * (t_cubed)

        return sigma

        for p in self.points:
            sigma += p[0] * (-t_cubed + 3)

    def calc_tween_points(self, accuracy):
        # self.render_points = np.zeros((accuracy, 2))
        self.render_points = []
        # TODO cap accuracy based on distance of points? Less are needed in some situations
        for i in range(accuracy+1):
            t = i/accuracy
            p = self.calc_point(t)
            self.render_points.append(pygame.math.Vector2(p[0], p[1]))

        return


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
            self.update_bernstein_points(self.points)

        return

# Drawing
    def draw_tween_lines(self, surface, color, width=1):
        num_points = len(self.render_points)
        p1 = self.render_points[0]
        p2 = p1
        for i in range(num_points-1):
            p1 = p2
            p2 = self.render_points[i+1]
            pygame.draw.line(surface, color, p1, p2)

        return

    def draw_control_points(self, surface, color, radius=1):
        for abs_point in self.abstract_points:
            abs_point.draw(surface, radius)
        return

    def draw_control_lines(self, surface, color):
        pygame.draw.line(surface, color, self.points[0], self.points[1])
        pygame.draw.line(surface, color, self.points[2], self.points[3])
        return

    def draw_render_points(self, surface, color, radius=1):
        for point in self.render_points:
            p = pygame.math.Vector2(point[0], point[1])
            pygame.draw.circle(surface, color, p, radius)


