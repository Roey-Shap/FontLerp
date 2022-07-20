
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import itertools
import sys
import time
import numpy as np
import custom_math
import copy

import global_variables as globvar
import toolbar
import ttfConverter
import fonts
import custom_colors as colors
import global_manager


import pygame
import curve
import contour
import glyph

pygame.init()
screen = globvar.screen
pygame.display.set_caption("Font Interpolater")
clock = globvar.clock
cursor = globvar.cursor

manager = global_manager.GlobalManager()

origin = globvar.origin
curves = globvar.curves
toolbar = toolbar.Toolbar()

manager.calculate_t_array()

line_width = 3
base_scale = 1

w, h = globvar.SCREEN_DIMENSIONS


# create default glyph bounding box
glyph_box = pygame.Rect(globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT, globvar.DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS)




# extracted_h1_data = ttfConverter.test_font_load("o", "AndikaNewBasic-B.ttf")
# extracted_h2_data = ttfConverter.test_font_load("o", "OpenSans-Light.ttf")
#
#
# test_fonts = ["AndikaNewBasic-B.ttf", "OpenSans-Light.ttf", "Calligraffiti.ttf"]
# test_glyphs = []
#
# for i, font in enumerate(test_fonts):
#     g = glyph.Glyph()
#     test_glyphs.append(g)
#     contours = ttfConverter.test_font_load("A", font)
#     num_curves = 0
#     for c in contours:
#         contour_object = ttfConverter.convert_quadratic_flagged_points_to_contour(c)
#         g.append_contour(contour_object)
#         num_curves += len(contour_object)
#         print(c)
#     g.set_offset(w * ((i+1)/5), h/2)
#     g.update_bounding_points()
#
#     print("Made glyph with upper left:", g.upper_left, "and lower right", g.lower_right)
#     r = g.lower_right - g.upper_left
#     print("... and thus width:", r[0], "and height:", r[1])
#     print("Glyph has", num_curves, "curves")
#     print("")


#
# test_h = glyph.Glyph()
# test_i = glyph.Glyph()
# for cnt in extracted_h1_data:
#     test_h.append_contour(ttfConverter.convert_quadratic_flagged_points_to_contour(cnt))
# # test_h.prune_curves()
# test_h.true_scale_by(0.07)
# test_h.set_offset(0, h/2)
#
# # test_i = test_h.copy()
# # test_i.em_scale(0.8)
#
# for cnt in extracted_h2_data:
#     test_i.append_contour(ttfConverter.convert_quadratic_flagged_points_to_contour(cnt))
# test_i.true_scale_by(0.1)
# test_i.set_offset(0, h/2)


# if len(test_h) == 2:
#     test_h.contours[-1].fill=contour.FILL.SUBTRACT


circle_const = globvar.CIRCLE_CONST
c1 = np.array([[0, 0],
               [0.1, 0.1],
               [0.2, 0.2],
               [0.25, 0.25]])
c2 = np.array([[0.25, 0.25],
               [0.35, 0.4],
               [0.4, 0.45],
               [0.5, 1]])
c3 = np.array([[0.5, 1],
               [0.125, 1],
               [-0.125, 1],
               [-0.5, 1]])
c4 = np.array([[-0.5, 1],
               [-0.625, 0.333],
               [-0.8, 0.667],
               [-1, 0]])
c5 = np.array([[-1, 0],
               [-0.77, -0.166],
               [-0.66, -0.333],
               [-0.5, -0.5]])
c6 = np.array([[-0.5, -0.5],
               [-0.166, -0.166],
               [-0.333, -0.333],
               [0, 0]])

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
                      [-1, 1+circle_const],
                      [-1, 1]])
circle_a4 = np.array([[-1, 1],
                      [-1, 1-circle_const],
                      [-circle_const, 0],
                      [0, 0]])
#
# cont = contour.Contour()
# cont.append_curves_from_np([c1, c2, c3, c4, c5, c6])
# cont.set_offset(4, 1.5)
# cont.set_scale(base_scale)

circle = contour.Contour()
circle.append_curves_from_np([circle_a1, circle_a2, circle_a3, circle_a4])
# circle.set_offset(4, 3.5)

circle_size = 100
circle_g = glyph.Glyph()
circle_g.append_contour(circle)
circle_g.worldspace_scale_by(circle_size)
circle_g.worldspace_offset(globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT)
circle_g.worldspace_offset(np.array([circle_size, 0], dtype=globvar.POINT_NP_DTYPE))

# circle_mini = circle.copy()
# circle_mini.fill=contour.FILL.SUBTRACT
# circle_mini.em_scale(0.7)
# circle_mini.em_offset(0.05, 0.25)
#
#
# circle_mini2 = circle_mini.copy()
# circle_mini2.fill = contour.FILL.ADD
# circle_mini2.em_scale(0.3)
# circle_mini2.em_offset(0, -1)
# circle_mini2.set_offset(4, 1.6)

#
# glyph_O = glyph.Glyph()
# glyph_O.append_contours_multi([circle, circle_mini])
# glyph_O.update_bounding_points()
#
#
# glyph_test = glyph.Glyph()
# glyph_test.append_contours_multi([circle_mini2, cont])
# glyph_test.update_bounding_points()


mixed_glyph = None
mappings = None
# mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)


# mixed_glyph = glyph.lerp_glyphs(test_h, test_i, mappings, 0)

not_scrolling = True

# Execution loop
running = True
while running:

    point_radius = 5

# Clock Updates
    clock.tick(globvar.FPS)
    globvar.mouse_scroll_directions = globvar.empty_offset
    prev_accuracy = globvar.BEZIER_ACCURACY
    last_not_scrolling = not_scrolling


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                globvar.KEY_SPACE_PRESSED = True
                # globvar.DEBUG = not globvar.DEBUG
            if event.key == pygame.K_r:
                print("Mapping disabled. Come uncomment this if you want a mapping.")
                # mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                globvar.KEY_SPACE_PRESSED = False
        elif event.type == pygame.MOUSEWHEEL:
            globvar.mouse_scroll_directions = np.array([event.x, event.y])

# Get Inputs

    # get mouse position relative to upper-left-hand corner
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = pygame.mouse.get_pressed()
    mouse_click_left = mouse_click[0] and not globvar.mouse_held
    mouse_held = mouse_click[0]
    mouse_scroll_directions = globvar.mouse_scroll_directions

    globvar.mouse_pos = mouse_pos
    globvar.mouse_click = mouse_click
    globvar.mouse_held = mouse_held
    globvar.mouse_click_left = mouse_click_left

    not_scrolling = np.all(np.isclose(globvar.mouse_scroll_directions, globvar.empty_offset))
    just_stopped_scrolling = not_scrolling and not last_not_scrolling

    # synchronize Cursor object with inputs
    cursor.update_mouse_variables()
    cursor.step(point_radius)
    panned_camera = cursor.panned_this_frame


    # if there were zoom updates, get update the bezier accuracy and get new t values
    if just_stopped_scrolling:
        manager.update_bezier_accuracy()
        if globvar.BEZIER_ACCURACY != prev_accuracy:
            manager.calculate_t_array()

    # update Bezier curves in response to any points which changed
    for curve in globvar.curves:
        # check for updates in abstract points (whose positions are in WORLDSPACE)
        curve.check_abstract_points()

        # update the curves based on the cursor's panning and offsetting
        if just_stopped_scrolling or panned_camera:
            curve.update_points()


    # Update zoom scale
    # for g in globvar.glyphs:
    #     g.em_scale(globvar.GLOBAL_SCALE * base_scale)

        # test_glyphs[0].set_scale(globvar.CAMERA_ZOOM * base_scale)

    glyph_box_offset_corner = globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT - globvar.CAMERA_OFFSET
    glyph_box = pygame.Rect(glyph_box_offset_corner * globvar.CAMERA_ZOOM,
                            globvar.DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS * globvar.CAMERA_ZOOM)


    # Reset the mixed contour on display
    if mixed_glyph is not None:
        mixed_glyph.destroy()
        mixed_glyph = None

        # mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)




    # Remix the glyph
    # mix_t = custom_math.map(np.sin(time.time()), -1, 1, 0, 1)
    # mixed_glyph = glyph.lerp_glyphs(test_h, test_i, mappings, mix_t)
    # mixed_glyph.set_offset(0, h/2)



    # Rendering ===============================================================================================
    screen.fill(colors.WHITE)

    # Left glyph unit boundaries
    border_width = 1
    corner_rad = 3
    pygame.draw.rect(screen, colors.BLACK, glyph_box, border_width, corner_rad, corner_rad, corner_rad, corner_rad)

    circle_g.draw(screen, point_radius)
    # test_glyphs[0].draw(screen, point_radius, [0, 0])
    # for g in test_glyphs:
    #     g.draw(screen, point_radius, [0, 0])

    # mixed_glyph.draw(screen, point_radius, [0, 0])
    # cont.draw_filled_polygon(screen, colors.RED)
    # cont.draw(screen, point_radius, color_gradient=True, width=line_width)
    # circle_mini2.draw_filled_polygon(screen, colors.BLUE)
    # circle_mini2.draw(screen, point_radius, color_gradient=True, width=line_width)

    # test_h.draw(screen, point_radius, [w/4, h/4])
    # test_i.draw(screen, point_radius, [w / 2, h / 4])
    # mixed_glyph.draw(screen, point_radius, [w/4, h/4])


    # test_contour.draw(screen, point_radius, color_gradient=True, width=line_width)

    # for c in globvar.contours:
    #     c.draw(screen, point_radius)


    # GUI drawing
    cursor.draw(screen)
    toolbar.draw(screen)


    # Debug drawing
    if globvar.DEBUG:
        pygame.draw.circle(screen, colors.RED, globvar.SCREEN_DIMENSIONS, 5)
        pygame.draw.circle(screen, colors.RED, origin, 5)

        debug_messages = ["Camera Offset: " + str(np.round(globvar.CAMERA_OFFSET, 4)),
                          "Global scale: " + str(round(globvar.CAMERA_ZOOM, 4)),
                          "Bezier Accuracy " + str(globvar.BEZIER_ACCURACY),
                          "Framerate: " + str(round(clock.get_fps(), 4)),
                          "Mouse position: " + str(mouse_pos),
                          "Mouse position in world: " + str(cursor.screen_to_world_space(mouse_pos)),
                          "Width, Height: " + str(globvar.SCREEN_DIMENSIONS),
                          "Mapping: " + str(mappings)]
        for i, message in enumerate(debug_messages):
            text_rect = pygame.Rect(0, 0, w/4, h)
            debug_text_surface = fonts.multiLineSurface(message, fonts.FONT_DEBUG, text_rect, colors.BLACK)
            # img = fonts.FONT_DEBUG.render(message, True, colors.BLACK)
            screen.blit(debug_text_surface, (20, (i+1) * 20))



    pygame.display.flip()






pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()