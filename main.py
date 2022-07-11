
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import itertools
import sys
import time
import numpy as np
import toolbar
import global_variables as globvar
import custom_colors as colors
import fonts

import pygame
import curve
import contour

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
# print(globvar.t_values)

w, h = globvar.SCREEN_DIMENSIONS

base_scale = 75

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

mixed_contour = None
mapping = contour.ofer_min(cont, circle)

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
    if mixed_contour is not None:
        mixed_contour.destroy()
        mixed_contour = None

        mapping = contour.ofer_min(cont, circle)


    # Remix the contour
    mix_t = (np.sin(time.time()) + 1) / 2
    mixed_contour = contour.lerp_contours_OMin(cont, circle, mapping, mix_t)
    mixed_contour.set_offset(6, 2.5)

    # update Bezier curves in response to any points which changed
    for curve in globvar.curves:
        # check for updates in abstract points
        curve.check_abstract_points(point_radius)
        curve.step()

    # Update zoom scale
    for c in globvar.contours:
        c.set_scale(globvar.global_scale * base_scale)


# Rendering
    screen.fill(colors.WHITE)

    cont.draw(screen, point_radius, color_gradient=True)
    circle.draw(screen, point_radius, color_gradient=True)
    mixed_contour.draw(screen, point_radius, color_gradient=True)
    # for c in globvar.contours:
    #     c.draw(screen, point_radius)

    # UI drawing
    cursor.draw(screen)
    toolbar.draw(screen)


    # Debug drawing
    if globvar.DEBUG:
        pygame.draw.circle(screen, colors.RED, globvar.SCREEN_DIMENSIONS, 5)
        pygame.draw.circle(screen, colors.RED, origin, 5)

        debug_messages = ["Framerate: " + str(round(clock.get_fps(), 4)),
                          "Width, Height: " + str(globvar.SCREEN_DIMENSIONS),
                          "Mapping: " + str(mapping)]
        for i, message in enumerate(debug_messages):
            text_rect = pygame.Rect(0, 0, w/4, h)
            debug_text_surface = fonts.multiLineSurface(message, fonts.FONT_DEBUG, text_rect, colors.BLACK)
            # img = fonts.FONT_DEBUG.render(message, True, colors.BLACK)
            screen.blit(debug_text_surface, (20, (i+1) * 20))



    pygame.display.flip()






pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()