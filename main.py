
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

globvar.t_values = np.array([[(i/globvar.bezier_accuracy)**3,
                              (i/globvar.bezier_accuracy)**2,
                              (i/globvar.bezier_accuracy), 1] for i in range(globvar.bezier_accuracy+1)], dtype=globvar.POINT_NP_DTYPE)
# print(globvar.t_values)

w, h = globvar.SCREEN_DIMENSIONS

size = 75

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
b = bezier.Bezier(c1 * size)
b2 = bezier.Bezier(c2 * size)
b3 = bezier.Bezier(c3 * size)
b4 = bezier.Bezier(c4 * size)
b5 = bezier.Bezier(c5 * size)
b6 = bezier.Bezier(c6 * size)

cont = contour.Contour()
cont.append_curve(b)
cont.append_curve(b2)
cont.append_curve(b3)
cont.append_curve(b4)
cont.append_curve(b5)
cont.append_curve(b6)
cont.offset(w/4, h/(1.75))
cont2 = cont.clone()
cont2.offset(0, -h/3)

circle = contour.Contour()
circle.append_curve(bezier.Bezier(circle_a1*size))
circle.append_curve(bezier.Bezier(circle_a2*size))
circle.append_curve(bezier.Bezier(circle_a3*size))
circle.append_curve(bezier.Bezier(circle_a4*size))

print(circle.is_closed())

circle.offset(w/4, h/2)

mapping = contour.ofer_min(cont, circle)
print(mapping)

# mix_t = (np.sin(time.time()) + 1) / 2
mixed_test = contour.lerp_contours_OMin(cont, circle, mapping, 0.5, debug_info=True)
mixed_test.offset(w / 4, -h/3)

mixed_contour = None

#
# b2 = bezier.Bezier(test_array * size)
# b2.offset(w/3, h/3)
#
# line = bezier.Line(np.array([[0, 0], [1, 1]])*size)
# line.offset(w/4, h/4)

# mixed_contour = contour.lerp_contours_OMin(cont, circle, mapping, 0)
# mixed_contour.offset(w / 8, 0)

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

    if mixed_contour is not None:
        mixed_contour.destroy()
        mixed_contour = None

    mix_t = (np.sin(time.time()) + 1) / 2
    mixed_contour = contour.lerp_contours_OMin(cont, circle, mapping, mix_t)
    mixed_contour.offset(w / 4, 0)

    # update Bezier curves in response to any points which changed
    for curve in curves:
        # check for updates in abstract points
        curve.check_abstract_points(point_radius)
        curve.step()

# Rendering
    screen.fill(colors.WHITE)


    for c in globvar.contours:
        c.draw(screen, point_radius)
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