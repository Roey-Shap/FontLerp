
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
import custom_colors as colors
import fonts
import ttfConverter

import pygame
import curve
import contour
import glyph

pygame.init()
screen = globvar.screen
pygame.display.set_caption("Font Interpolater")
clock = globvar.clock
cursor = globvar.cursor


origin = globvar.origin
curves = globvar.curves
toolbar = toolbar.Toolbar()

globvar.t_values = np.array([[(i/globvar.bezier_accuracy)**3,
                              (i/globvar.bezier_accuracy)**2,
                              (i/globvar.bezier_accuracy), 1] for i in range(globvar.bezier_accuracy+1)], dtype=globvar.POINT_NP_DTYPE)

line_width = 3
w, h = globvar.SCREEN_DIMENSIONS

base_scale = 75




extracted_h1_data = ttfConverter.test_font_load("o", "AndikaNewBasic-B.ttf")
extracted_h2_data = ttfConverter.test_font_load("o", "OpenSans-Light.ttf")
test_h = glyph.Glyph()
test_i = glyph.Glyph()
for cnt in extracted_h1_data:
    test_h.append_contour(ttfConverter.convert_quadratic_flagged_points_to_contour(cnt))
# test_h.prune_curves()
test_h.em_scale(0.07)
test_h.set_offset(0, h/2)

# test_i = test_h.copy()
# test_i.em_scale(0.8)

for cnt in extracted_h2_data:
    test_i.append_contour(ttfConverter.convert_quadratic_flagged_points_to_contour(cnt))
test_i.em_scale(0.1)
test_i.set_offset(0, h/2)


if len(test_h) == 2:
    test_h.contours[-1].fill=contour.FILL.SUBTRACT

print("# Contours:", len(test_h))


circle_const = 0.5522847
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


cont = contour.Contour()
cont.append_curves_from_np([c1, c2, c3, c4, c5, c6])
cont.set_offset(4, 1.5)
cont.set_scale(base_scale)

circle = contour.Contour()
circle.append_curves_from_np([circle_a1, circle_a2, circle_a3, circle_a4])
circle.set_offset(4, 3.5)
circle.set_scale(base_scale)

circle_mini = circle.copy()
circle_mini.fill=contour.FILL.SUBTRACT
circle_mini.em_scale(0.7)
circle_mini.em_offset(0.05, 0.25)

circle_mini2 = circle_mini.copy()
circle_mini2.fill = contour.FILL.ADD
circle_mini2.em_scale(0.3)
circle_mini2.em_offset(0, -1)
circle_mini2.set_offset(4, 1.6)


glyph_O = glyph.Glyph()
glyph_O.append_contours_multi([circle, circle_mini])
glyph_O.update_bounding_points()


glyph_test = glyph.Glyph()
glyph_test.append_contours_multi([circle_mini2, cont])
glyph_test.update_bounding_points()

# mixed_contour = None
# mapping, mapping_score = contour.find_ofer_min_mapping(cont, circle)

mixed_glyph = None
mappings = None
mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)

# test_cont = contour.Contour()
# test_cont.append_curves_from_np([c6[0:3]])

mixed_glyph = glyph.lerp_glyphs(test_h, test_i, mappings, 0)

# Execution loop
running = True
while running:

    point_radius = 3

# Clock Updates
    clock.tick(globvar.FPS)

    globvar.mouse_scroll_directions = globvar.empty_offset

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                globvar.DEBUG = not globvar.DEBUG
            if event.key == pygame.K_r:
                mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)
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

    # synchronize Cursor object with inputs
    cursor.update_mouse_variables()
    cursor.step(point_radius)


    # Update zoom and panning
    globvar.global_scale += mouse_scroll_directions * globvar.scroll_delta


    # Reset the mixed contour on display
    if mixed_glyph is not None:
        mixed_glyph.destroy()
        mixed_glyph = None

        # mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)




    # Remix the glyph
    mix_t = custom_math.map(np.sin(time.time()), -1, 1, 0, 1)
    mixed_glyph = glyph.lerp_glyphs(test_h, test_i, mappings, mix_t)
    mixed_glyph.set_offset(0, h/2)



    # update Bezier curves in response to any points which changed
    for curve in globvar.curves:
        # check for updates in abstract points
        curve.check_abstract_points(point_radius)
        curve.step()

    # Update zoom scale
    # for c in globvar.contours:
    #     c.set_scale(globvar.global_scale * base_scale) <---------------- Important! TODO


# Rendering
    screen.fill(colors.WHITE)

    # mixed_glyph.draw(screen, point_radius, [0, 0])
    # cont.draw_filled_polygon(screen, colors.RED)
    # cont.draw(screen, point_radius, color_gradient=True, width=line_width)
    # circle_mini2.draw_filled_polygon(screen, colors.BLUE)
    # circle_mini2.draw(screen, point_radius, color_gradient=True, width=line_width)

    # test_h.draw(screen, point_radius, [w/4, h/4])
    # test_i.draw(screen, point_radius, [w / 2, h / 4])
    mixed_glyph.draw(screen, point_radius, [w/4, h/4])


    # test_contour.draw(screen, point_radius, color_gradient=True, width=line_width)

    # test_cont.set_offset(3, 3)ou
    # test_cont.set_scale(globvar.global_scale * base_scale)

    # glyph_O.draw(screen, point_radius, [0, 0])
    # glyph_test.draw(screen, point_radius, [w/2, 0])
    # circle.draw_filled_polygon(screen, colors.BLUE)
    # circle_mini.draw_filled_polygon(screen, colors.BLUE)
    # circle.draw(screen, point_radius, color_gradient=True, width=line_width)
    # circle_mini.draw(screen, point_radius, color_gradient=True, width=line_width)
    # gray_value = 0.8 * 255
    # fill_color = (gray_value, gray_value, gray_value)
    # cont.draw_filled_polygon(screen, fill_color)
    # cont.draw(screen, point_radius, color_gradient=True, width=line_width)
    # circle.draw(screen, point_radius, color_gradient=True, width=line_width)
    # mixed_contour.draw(screen, point_radius, color_gradient=True, width=line_width)
    # for c in globvar.contours:
    #     c.draw(screen, point_radius)

    # UI drawing
    cursor.draw(screen)
    toolbar.draw(screen)


    # Debug drawing
    if globvar.DEBUG:
        pygame.draw.circle(screen, colors.RED, globvar.SCREEN_DIMENSIONS, 5)
        pygame.draw.circle(screen, colors.RED, origin, 5)

        debug_messages = ["Press 'R' to pick a new random mapping",
                          "Framerate: " + str(round(clock.get_fps(), 4)),
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