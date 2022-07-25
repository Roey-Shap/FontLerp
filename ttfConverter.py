"""
https://stackoverflow.com/questions/40437308/retrieving-bounding-box-and-bezier-curve-data-for-all-glyphs-in-a-ttf-font-file
"""

import ttfquery
import ttfquery.glyph
import curve
import contour
import numpy as np
import global_variables as globvar
import glyph

def test_font_load(char, ttf_file_name):
    # print("Font import information")

    font_url = "Test_Fonts/" + ttf_file_name
    font = ttfquery.describe.openFont(font_url)
    g = ttfquery.glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
    g_contours = g.calculateContours(font)

# from http://ttfquery.sourceforge.net/_modules/ttfquery/glyph.html#integrateQuadratic
#############
    expanded_contours = []
    for contour in g_contours:
        if len(contour) < 3:
            return ()
        _set = contour[:]

        def on(point):
            """Is this record on the contour?"""
            (Ax, Ay), Af = point
            return Af == 1

        def merge(p1, p2):
            """Merge two off-point records into an on-point record"""
            (Ax, Ay), Af = p1
            (Bx, By), Bf = p2
            return ((Ax + Bx) / 2.0, (Ay + By) / 2.0), 1

        # create an expanded set so that all adjacent
        # off-curve items have an on-curve item added
        # in between them
        last = contour[-1]
        expanded = []
        for item in _set:
            if (not on(item)) and (not on(last)):
                expanded.append(merge(last, item))
            expanded.append(item)
            last = item
        # result = []
        # last = expanded[-1]
        # print(expanded)
        # print("... done!")

        expanded_contours.append(expanded)
    return expanded_contours
##########################
    # g.compile(font)
    #
    # glyf = font['glyf']
    # charglyf = glyf[ttfquery.glyphquery.glyphName(font, char)]
    # print(charglyf.endPtsOfContours)
    # print("===")
    # print(charglyf.coordinates)
    # for outline in g.outlines:
    #     print(outline)
    # contours = g.calculateContours(font)
    # print("Contours:", len(contours))
    # for contour in contours:
    #     for point, flag in contour:
    #         print(point, flag)


def flip_y(p):
    return [p[0], -p[1]]

def convert_quadratic_flagged_points_to_contour(flagged_points):
    # loop until no "1" flag is found; if two consecutive "1"s are found, make a point midway between them
    last_endpoint = flagged_points[0][0]
    current_points = [flip_y(last_endpoint)]
    cont = contour.Contour()
    # start from index 1 since we already have the first startpoint in 'current_points'
    for point_index in range(1, len(flagged_points)):
        current_coords, is_endpoint = flagged_points[point_index]
        current_coords = flip_y(current_coords)
        current_points.append(current_coords)
        if is_endpoint:
            if len(current_points) == 4:
                print("Cubic!")
            if len(current_points) == 2:
                midway = (0.5*(current_points[0][0] + current_points[1][0]), 0.5*(current_points[0][1] + current_points[1][1]))
                current_points = [current_points[0], midway, current_points[1]]
            quad_curve = curve.Bezier(np.array(current_points, dtype=globvar.POINT_NP_DTYPE))
            # print("Making quadratic bezier from points:", np.array(current_points))
            cont.append_curve(quad_curve)
            current_points = [current_coords]                           # reset the list for the next curve

    cont.update_bounds()
    return cont

def glyph_from_font(char, font_file_name):
    g = glyph.Glyph(char)
    extracted_font_data = test_font_load(char, font_file_name)
    for cnt in extracted_font_data:
        formatted_contour = convert_quadratic_flagged_points_to_contour(cnt)
        formatted_contour.update_bounds()
        g.append_contour(formatted_contour)
        g.update_bounds()

    g.sort_contours_by_fill()
    return g