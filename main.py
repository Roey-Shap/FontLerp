
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import sys
import pygame
import global_variables as globvar
import custom_colors as colors
import bezier
import numpy as np

pygame.init()
screen = globvar.screen
pygame.display.set_caption("Font Interpolater")
clock = globvar.clock

origin = globvar.origin

Beziers = []

size = 80
offset = 150
test_array = [[size, size],
              [size, size*2],
              [size*3, size*2],
              [size*3, size]]
b = bezier.BezierCubic2D(np.array(test_array))

b2 = b.copy()
b2.offset(offset, offset)


Beziers.append(b)
Beziers.append(b2)

bezier_accuracy = 15

# Execution loop
running = True
while running:

    point_radius = 4

# Clock Updates
    clock.tick(globvar.FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False



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


    for bezier in Beziers:
        # check for updates in abstract points
        bezier.check_abstract_points(point_radius)
        bezier.calc_tween_points(bezier_accuracy)

# Rendering
    screen.fill(colors.WHITE)

    for bezier in Beziers:
        bezier.draw_control_lines(screen, colors.LT_GRAY)
        bezier.draw_tween_lines(screen, colors.BLACK)
        bezier.draw_control_points(screen, colors.LT_GRAY, radius=point_radius)

    #
    # pygame.draw.circle(screen, colors.RED, q, 3)
    # pygame.draw.circle(screen, colors.RED, origin, 3)


    pygame.display.flip()






pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()