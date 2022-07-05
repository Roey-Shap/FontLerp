
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import sys
import pygame
import global_variables as globvar
import custom_colors as colors
import bezier
import numpy as np
import toolbar

pygame.init()
screen = globvar.screen
pygame.display.set_caption("Font Interpolater")
clock = globvar.clock
cursor = globvar.cursor


origin = globvar.origin
curves = globvar.curves
toolbar = toolbar.Toolbar()


size = 80
offset = 80
test_array = [[size, size],
              [size, size*2],
              [size*3, size*2],
              [size*3, size]]
b = bezier.Bezier(np.array(test_array))

b2 = b.copy()
b2.offset(offset, offset)

b3 = bezier.Bezier(np.array(test_array[0:3]))
b3.offset(offset*2, offset*2)
b4 = bezier.Line(np.array(test_array[0:2]))
b4.offset(offset*2, offset)

curves.append(b)
curves.append(b2)
curves.append(b3)
curves.append(b4)

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

    for curve in curves:
        curve.draw(screen, point_radius)

    # UI drawing
    cursor.draw(screen)
    toolbar.draw(screen)


    # Debug drawing
    pygame.draw.circle(screen, colors.RED, globvar.SCREEN_DIMENSIONS, 5)
    pygame.draw.circle(screen, colors.RED, origin, 5)


    pygame.display.flip()






pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()