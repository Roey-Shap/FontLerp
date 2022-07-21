import global_variables as globvar
import numpy as np
import pygame
import custom_colors

class GlobalManager(object):
    class State(int):
        NO_TOOL = 0
        ADJUSTING_POINTS = 1
        ADDING_CURVES = 2
        MAPPING_CURVES = 3

    class MappingCurveState(int):
        OMIN = 0
        INSERTION = 1


    def __init__(self):
        return

    """
    Based on the level of zoom, adjust the accuracy with which Bezier curves are drawn.
    """
    def update_bezier_accuracy(self):
        globvar.BEZIER_ACCURACY = int(2.5 * np.log2(globvar.CAMERA_ZOOM * 30)) + 1
        return

    def calculate_t_array(self):
        globvar.t_values = np.array([[(i / globvar.BEZIER_ACCURACY) ** 3,
                                      (i / globvar.BEZIER_ACCURACY) ** 2,
                                      (i / globvar.BEZIER_ACCURACY), 1] for i in range(globvar.BEZIER_ACCURACY + 1)],
                                    dtype=globvar.POINT_NP_DTYPE)
        return


    """
    Draw lines between each of the beziers based on the mappings provided
    """
    def draw_mapping_reduction(self, surface, glyph1, glyph2, mappings):
        n1 = len(glyph1)
        n2 = len(glyph2)

        if n2 > n1:
            raise AttributeError("Glyph 1 needs to have at least as many curves as Glyph 2 to draw the correct mapping")

        start_color = custom_colors.NEON_BLUE
        end_color = custom_colors.NEON_RED
        total_mappings = len(mappings)
        # iterate through each contour of G1 and show the curve that it's mapped to in G2
        for g1_contour_index, g1_mapping_details in enumerate(mappings):
            current_mapped_contours_color = custom_colors.mix_color(start_color, end_color, g1_contour_index / total_mappings)
            if g1_contour_index == total_mappings - 1:
                current_mapped_contours_color = end_color

            g2_contour_index, g2_contour_mapping = g1_mapping_details
            g1_contour = glyph1.contours[g1_contour_index]
            g2_contour = glyph2.contours[g2_contour_index]

            # take each pair in the mapping; if it's not a null, draw a line
            for pair in g2_contour_mapping.items():
                c1_index, c2_index = pair
                if c2_index is not None:
                    c1_curve_center = g1_contour.curves[c1_index].get_center_camera()
                    c2_curve_center = g2_contour.curves[c2_index].get_center_camera()
                    print(c1_curve_center)
                    pygame.draw.line(surface, current_mapped_contours_color, c1_curve_center, c2_curve_center)

