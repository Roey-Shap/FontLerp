import numpy as np
import pygame
import custom_colors
import global_variables as globvar
import global_manager
import point as AbsPoint
import copy
import custom_math

""" 
Represents a 2D Bezier curve
Is comprised of 3 or 4 control points: p0, p1, p2, p3
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
    def __init__(self, worldspace_points):
        if not isinstance(worldspace_points, np.ndarray):
            raise ValueError("tried inputting: \'", worldspace_points, "\' as an ndarray when it wasn't one")

        worldspace_points = worldspace_points.astype(globvar.POINT_NP_DTYPE)

        self.num_points = worldspace_points.shape[0]

        self.worldspace_points = worldspace_points.copy()
        self.cameraspace_points = worldspace_points.copy()
        self.unoffset_cameraspace_points = worldspace_points.copy()

        self.abstract_points = []
        self.average_point = None

        self.parent_glyph_upper_left = globvar.empty_offset.copy()

        self.current_contour_relative_angle = None

        for i, point in enumerate(self.cameraspace_points):
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

    """
    offset this curve's points in WORLDSPACE
    """
    def worldspace_offset_by(self, offset):
        self.worldspace_points += offset
        self.update_points()
        return

    """
    offset this curve's points in WORLDSPACE
    """
    def worldspace_scale_by(self, scale):
        self.worldspace_points *= scale
        self.update_points()
        return

    def get_dimensions_world(self):
        return self.get_lower_right_world() - self.get_upper_left_world()

    def get_upper_left_world(self):
        return np.min(self.worldspace_points, axis=0)

    def get_lower_right_world(self):
        return np.max(self.worldspace_points, axis=0)

    def get_dimensions_camera(self):
        return self.get_lower_right_camera() - self.get_upper_left_camera()

    def get_upper_left_camera(self):
        return np.min(self.cameraspace_points, axis=0)

    def get_lower_right_camera(self):
        return np.max(self.cameraspace_points, axis=0)

    def get_center_camera(self):
        return np.average(self.cameraspace_points, axis=0)

    """
    Return the angle of line formed by the endpoints in degrees
    """
    def get_angle(self):
        p1 = self.cameraspace_points[0]
        p2 = self.cameraspace_points[-1]
        difference_vector = p1-p2
        return np.arctan(difference_vector[1] / difference_vector[0]) * 180 / np.pi

# Point Manipulation
    """
    For each point, check if the mouse is hovering over it
    """
    def check_abstract_points(self):
        change_detected = False
        for i, abs_point in enumerate(self.abstract_points):
            current_change = abs_point.changed_position_this_frame
            change_detected = change_detected or current_change
            if current_change:
                self.worldspace_points[i] = (abs_point.np_coords()/globvar.CAMERA_ZOOM) + globvar.CAMERA_OFFSET
        # if change_detected:
        #     # a change in some point was detected; update cameraspace points, tween points, etc.
        #     # could change just the potentially two curves affected, but I think it's fine.
        #     self.update_points()

        return change_detected

    """
    Draw the "control points" which influence the Curve's shape in between the endpoints
    """
    def draw_control_points(self, surface, radius=1):
        for abs_point in self.abstract_points:
            abs_point.draw(surface, radius)
        return


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
    def __init__(self, worldspace_points):
        super().__init__(worldspace_points)

        if worldspace_points.shape[0] < 3 or worldspace_points.shape[0] > 4:
            raise ValueError("Beziers must have 3 or 4 points: matrix", worldspace_points,
                             "was input with:", worldspace_points.shape[0], "points")

        self.adjustment_matrix = Bezier.adjustment_matrices[self.num_points-3]

        self.bernstein_points = None
        self.tween_points = None
        self.worldspace_tween_points = None
        self.unoffset_tween_points = None


        self.worldspace_points = worldspace_points
        self.update_points()
        return


    def get_length_world(self):
        sigma = 0
        num_render_points = len(self.worldspace_tween_points)
        for i in range(num_render_points-1):
            p1 = self.worldspace_tween_points[i]
            p2 = self.worldspace_tween_points[i+1]
            sigma += np.linalg.norm(p1-p2)

        return sigma


    def get_length_camera(self):
        sigma = 0
        num_render_points = len(self.tween_points)
        for i in range(num_render_points-1):
            p1 = self.tween_points[i]
            p2 = self.tween_points[i+1]
            sigma += np.linalg.norm(p1-p2)

        return sigma


    """
    Check against all of the current worldspace tween points; if it's close enough to one of them, it's on
    """
    def contains_point(self, test_point):
        for point in self.worldspace_tween_points:
            if np.all(np.isclose(point, test_point)):
                return True

        return False

    """
    Sets this Bezier's points and caches linear combinations of them for interpolated points.
    Returns a (2x4) matrix
    ===
    the adjusted Berstein polynomial form above yields 
    an inner product of (a vector of powers of t) and (linear combinations of the control points)
    cache those constant linear combinations of the control points for calculating points in between 
    """
    def update_points(self):
        self.cameraspace_points = (self.worldspace_points - globvar.CAMERA_OFFSET) * globvar.CAMERA_ZOOM
        self.unoffset_camaraspace_points = self.cameraspace_points - self.parent_glyph_upper_left
        for i, abstract_point in enumerate(self.abstract_points):
            abstract_point.update_coords(self.cameraspace_points[i])

        # turn pretty (nx2) points into (2xn) via transpose
        # (2xn) = (2xn)*(nxn)
        self.bernstein_points = np.matmul(self.cameraspace_points.T, self.adjustment_matrix)
        self.calc_tween_points()
        self.unoffset_tween_points = self.tween_points - self.parent_glyph_upper_left
        self.worldspace_tween_points = custom_math.camera_to_worldspace(self.tween_points)

        # self.average_point = np.average(self.cameraspace_points, axis=0)
        return

    """
    Calculates a 2D point resulting from the weighted interpolation of the control points.
    Returns a (2x1) matrix (a single point)
    """
    def calc_tween_points(self):
        size_adj = 0 if self.num_points == 4 else 1
        self.tween_points = np.matmul(self.bernstein_points, globvar.t_values[:, size_adj:4].T).T
        return

# Drawing
    def draw(self, surface, point_radius, input_colors=None, width=1, flush_with_origin=True):
        colors = [custom_colors.LT_GRAY, custom_colors.BLACK, custom_colors.BLUE]
        if input_colors is not None:
            colors = input_colors
        self.draw_tween_lines(surface, colors[1], width=width, flush_with_origin=flush_with_origin)

        if globvar.show_extra_curve_information:
            self.draw_control_lines(surface, colors[0], width=width, flush_with_origin=flush_with_origin)
        return


    """
    Draws the lines connected the precomputed "render points
    """
    def draw_tween_lines(self, surface, color, width=1, flush_with_origin=False):
        conditional_points = self.unoffset_tween_points if flush_with_origin else self.tween_points
        pygame.draw.aalines(surface, color, closed=False, points=conditional_points)
        return

    """
    Draw the lines from the control points to show how they influence the Bezier's curvature
    """
    def draw_control_lines(self, surface, color, width=1, flush_with_origin=False):
        width = round(width/3)
        conditional_points = self.unoffset_camaraspace_points if flush_with_origin else self.cameraspace_points
        pygame.draw.line(surface, color, conditional_points[0], conditional_points[1], width=width)
        pygame.draw.line(surface, color, conditional_points[-2], conditional_points[-1], width=width)
        return

    """
    Draws the precomputed "render points" which approximate the Bezier 
    """
    def draw_render_points(self, surface, color, radius=1):
        for point in self.tween_points:
            p = pygame.math.Vector2(point[0], point[1])
            pygame.draw.circle(surface, color, p, radius)

        return

    """ 
    from https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
    Split the 
    """
    def de_casteljau(self, t):
        if self.num_points == 3:
            return self.de_casteljau_quad(t)
        elif self.num_points == 4:
            return self.de_casteljau_cubic(t)


    def de_casteljau_quad(self, t):
        A, B, C = self.worldspace_points
        E = custom_math.interpolate_np(A, B, t)
        F = custom_math.interpolate_np(B, C, t)

        G = custom_math.interpolate_np(E, F, t)
        quad1 = Bezier(np.array([A, E, G]))
        quad2 = Bezier(np.array([G, F, C]))

        return quad1, quad2

    # TODO Implement Cubic splitting - maybe even also quad -> cubic split? We'll see if it's still useful...
    def de_casteljau_cubic(self, t):
        raise ValueError("UNIMPLEMENTED")
        return


        # beta = [c for c in self.worldspace_points]  # values in this list are overridden
        # degree = len(beta)
        # for j in range(1, degree):
        #     for k in range(degree - j):
        #         beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
        # return beta[0]



