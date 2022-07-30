
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

print("NOTE: THERE'S STILL A LINGERING QUESTION OF HOW TO MAP CONTOURS OF DIFFERENT FILL TYPES. PLAY"
      "AROUND WITH SOME TEXT TO SEE WHAT I MEAN. THERE'S PROBABLY A SIMPLE SOLUTION BUT I DON'T"
      "HAVE THE ENERGY OR TIME RIGHT NOW.")
print("")
print("IN ADDITION, THE RELATIVE PROJECTION METHOD STILL NEEDS SOME WORK - CODED IN ONE GO AND SOMETIMES"
      "DOESN'T HAVE TWO CURVES IN ONE MAPPING SLOT... :(")

manager = global_manager.GlobalManager()

screen = globvar.screen
clock = globvar.clock
cursor = globvar.cursor

origin = globvar.origin
curves = globvar.curves
toolbar = globvar.toolbar

line_width = 3

w, h = globvar.SCREEN_DIMENSIONS
global_manager.set_mapping_and_lerping_methods("Pillow Projection")


# create default glyph bounding box
glyph_box = pygame.Rect(globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT, globvar.DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS)


# extracted_h1_data = ttfConverter.test_font_load("o", "AndikaNewBasic-B.ttf")
# extracted_h2_data = ttfConverter.test_font_load("o", "OpenSans-Light.ttf")
#

test_fonts = ["AndikaNewBasic-B.ttf", "OpenSans-Light.ttf", "Calligraffiti.ttf"]

test_text = "More substantial text. This is a short poem written to test the wrapping capabilities."
font1 = "Alef-Regular.ttf"
font2 = "AndikaNewBasic-B.ttf"
font_cursive = "Calligraffiti.ttf"
#

test_lerped_glyphs = None
# test_lerped_glyphs = global_manager.get_glyphs_from_text(test_text, font1, font2,
#                                                          wrap_x=w)

test_o = ttfConverter.glyph_from_font("q", font2)
test_o.worldspace_scale_by(globvar.EM_TO_FONT_SCALE)
test_o.update_bounds()
up_left_origin_diff = test_o.get_upper_left_world()
test_o.worldspace_offset_by(-up_left_origin_diff)
test_o.update_bounds()

test_glyphs = []



character = "p"
extracted_h1_data = ttfConverter.test_font_load(character, font1)
extracted_h2_data = ttfConverter.test_font_load(character, font_cursive)
test_h = glyph.Glyph(character)
test_i = glyph.Glyph(character)
for cnt in extracted_h1_data:
    formatted_contour = ttfConverter.convert_quadratic_flagged_points_to_contour(cnt)
    test_h.append_contour(formatted_contour)
# test_h.prune_curves()
test_h.worldspace_scale_by(0.1)
test_h.worldspace_offset_by(np.array([w/4, h/2]))
test_h.update_bounds()
test_h.sort_contours_by_fill()
# test_h.worldspace_offset_by(-test_h.get_center_world())


for cnt in extracted_h2_data:
    formatted_contour = ttfConverter.convert_quadratic_flagged_points_to_contour(cnt)
    test_i.append_contour(formatted_contour)

test_i.worldspace_scale_by(0.15)
test_i.worldspace_offset_by(np.array([w*3/4, h/2]))
test_i.update_bounds()
test_i.sort_contours_by_fill()


global_manager.set_active_glyphs(test_h, test_i)

# CIRCLE
# circle = contour.get_unit_circle_contour()
# circle_size = 100
# circle_g = glyph.Glyph()
# circle_g.append_contour(circle)
# circle_g.worldspace_scale_by(circle_size)
# circle_g.worldspace_offset_by(globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT)
# circle_g.worldspace_offset_by(np.array([circle_size, 0], dtype=globvar.POINT_NP_DTYPE))


mixed_glyph = None
mappings = None
global_manager.make_mapping_from_active_glyphs()


not_scrolling = True

# Execution loop
running = True
while running:

    point_radius = globvar.POINT_DRAW_RADIUS

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

    manager.step()
    panned_camera = cursor.panned_this_frame


    # if there were zoom updates, get update the bezier accuracy and get new t values
    if just_stopped_scrolling:
        global_manager.update_bezier_accuracy()
        if globvar.BEZIER_ACCURACY != prev_accuracy:
            global_manager.calculate_t_array()

    # update Bezier curves in response to any points which changed
    for curve in globvar.curves:
        # check for updates in abstract points (whose positions are in WORLDSPACE)
        curve.check_abstract_points()

        # update the curves based on the cursor's panning and offsetting
        if just_stopped_scrolling or panned_camera:
            curve.update_points()

    glyph_box_offset_corner = globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT - globvar.CAMERA_OFFSET
    glyph_box = pygame.Rect(glyph_box_offset_corner * globvar.CAMERA_ZOOM,
                            globvar.DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS * globvar.CAMERA_ZOOM)


    # Reset the mixed contour on display
    if globvar.lerped_glyph is not None:
        print("Destroying previous lerped_glyph")
        globvar.lerped_glyph.destroy()
        globvar.lerped_glyph = None

    for g in globvar.glyphs:
        g.update_bounds()


    # Remix the glyph
    if globvar.current_glyph_mapping_is_valid and globvar.show_mixed_glyph:
        mix_t = custom_math.map(np.sin(time.time()), -1, 1, 0, 1)
        global_manager.lerp_active_glyphs(mix_t)


    # Rendering ===============================================================================================
    if globvar.update_screen:
        screen.fill(colors.WHITE)

    # Left glyph unit boundaries
    border_width = 1
    corner_rad = 3
    pygame.draw.rect(screen, colors.BLACK, glyph_box, border_width, corner_rad, corner_rad, corner_rad, corner_rad)

    if globvar.lerped_glyph is not None and globvar.show_mixed_glyph:
        globvar.lerped_glyph.draw(screen, point_radius)


    # test_h.draw(screen, point_radius)
    # test_i.draw(screen, point_radius)

    if test_lerped_glyphs is not None:
        global_manager.draw_lerped_text(screen, test_lerped_glyphs)

    for debug_point in globvar.debug_width_points:
        pygame.draw.circle(screen, colors.BLUE, custom_math.world_to_cameraspace(np.array(debug_point)), 5)

    if globvar.show_current_glyph_mapping:
        manager.draw_mapping_pillow_projection(screen, mappings)


    if globvar.update_screen:
        # GUI drawing
        cursor.draw(screen)
        toolbar.draw(screen)

    test_o.draw(screen, globvar.POINT_DRAW_RADIUS)

    # Debug drawing
    if globvar.update_screen: # TODO globvar.DEBUG and
        # pygame.draw.circle(screen, colors.BLACK, custom_math.world_to_cameraspace(globvar.SCREEN_DIMENSIONS), 5)
        # pygame.draw.circle(screen, colors.BLACK, custom_math.world_to_cameraspace(origin), 5)

        debug_messages = ["Camera Offset: " + str(np.round(globvar.CAMERA_OFFSET, 4)),
                          "Global scale: " + str(round(globvar.CAMERA_ZOOM, 4)),
                          "Bezier Accuracy " + str(globvar.BEZIER_ACCURACY),
                          "Mouse position: " + str(np.round(mouse_pos, 3)),
                          "Mouse position in world: " + str(np.round(cursor.screen_to_world_space(mouse_pos), 3)),
                          "Width, Height: " + str(globvar.SCREEN_DIMENSIONS),
                          "Global Manager State: " + str(manager.state)]
                          # "All Mappings: " + str(globvar.glyph_mappings)]
                          # "Glyph Mapping: " + str(globvar.current_glyph_mapping)]
        for i, message in enumerate(debug_messages):
            text_rect = pygame.Rect(0, 0, w/4, h)
            debug_text_surface = fonts.multiLineSurface(message, fonts.FONT_DEBUG, text_rect, colors.BLACK)
            # img = fonts.FONT_DEBUG.render(message, True, colors.BLACK)
            screen.blit(debug_text_surface, (20, (i+1) * 20))


    if globvar.update_screen:
        pygame.display.flip()

    # globvar.update_screen = False




pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()