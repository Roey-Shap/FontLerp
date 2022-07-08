
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
import bezier
import contour

pygame.init()
screen = globvar.screen
pygame.display.set_caption("Font Interpolater")
clock = globvar.clock
cursor = globvar.cursor


origin = globvar.origin
curves = globvar.curves
toolbar = toolbar.Toolbar()


w, h = globvar.SCREEN_DIMENSIONS

size = 50

circle_const = 0.5522847
c1 = np.array([[0, 0],
               [0.25, 0],
               [0.5, 0.5],
               [1, 1]])
c2 = np.array([[1, 1],
               [0.5, 2],
               [0, 1],
               [0, 0]])
circle_a1 = np.array([[0, 0],
                      [circle_const, 0],
                      [1, 1 - circle_const],
                      [1, 1]])
circle_a2 = np.array([[1, 1],
                      [1, 1 + circle_const],
                      [circle_const, 2],
                      [0, 2]])
b = bezier.Bezier(c1 * size)
b2 = bezier.Bezier(c2 * size)

cont = contour.Contour()
cont.append_curve(b)
cont.append_curve(b2)
cont.offset(w/4, h/2)

circle = contour.Contour()
circle.append_curve(bezier.Bezier(circle_a1*size))
circle.append_curve(bezier.Bezier(circle_a2*size))
circle.append_curve(bezier.Bezier(np.array([[-row[0], row[1]] for row in circle_a1])*size))
circle.append_curve(bezier.Bezier(np.array([[-row[0], row[1]] for row in circle_a2])*size))

circle.offset(w*3/4, h/2)

mapping = contour.ofer_min(circle, cont)
mixed = contour.lerp_contours_OMin(circle, cont, mapping, 0.5)
mixed.offset(w*1/4, -h/4)

#
# b2 = bezier.Bezier(test_array * size)
# b2.offset(w/3, h/3)
#
# line = bezier.Line(np.array([[0, 0], [1, 1]])*size)
# line.offset(w/4, h/4)


# Execution loop
running = True
while running:

    point_radius = 3

# Clock Updates
    clock.tick(globvar.FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                globvar.DEBUG = not globvar.DEBUG



# Get Inputs

    # get mouse position relative to upper-left-hand corner
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = pygame.mouse.get_pressed()
    mouse_click_left = mouse_click[0] and not globvar.mouse_held
    mouse_held = mouse_click[0]

    globvar.mouse_pos = mouse_pos
    globvar.mouse_click = mouse_click
    globvar.mouse_held = mouse_held
    globvar.mouse_click_left = mouse_click_left

    # synchronize Cursor object with inputs
    cursor.update_mouse_variables()
    cursor.step(point_radius)

    # update Bezier curves in response to any points which changed
    for curve in curves:
        # check for updates in abstract points
        curve.check_abstract_points(point_radius)
        curve.step()

# Rendering
    screen.fill(colors.WHITE)


    for contour in globvar.contours:
        contour.draw(screen, point_radius)
    # for curve in curves:
    #     curve.draw(screen, point_radius)

    # UI drawing
    cursor.draw(screen)
    toolbar.draw(screen)


    # Debug drawing
    if globvar.DEBUG:
        pygame.draw.circle(screen, colors.RED, globvar.SCREEN_DIMENSIONS, 5)
        pygame.draw.circle(screen, colors.RED, origin, 5)
        debug_message = str(round(clock.get_fps(), 4)) + ", Width, Height: " + str(globvar.SCREEN_DIMENSIONS)
        img = fonts.TEST.render(debug_message, True, colors.BLACK)
        screen.blit(img, (20, 20))


    pygame.display.flip()






pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()