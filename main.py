import sys
import time
import numpy as np
import custom_math

import custom_pygame
import global_variables as globvar
import ptext
import ttfConverter
import fonts
import custom_colors as colors
import global_manager

import pygame

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

line_width = globvar.LINE_THICKNESS

w, h = globvar.SCREEN_DIMENSIONS
global_manager.set_mapping_and_lerping_methods("Pillow Projection")

# create default glyph bounding box
glyph_box = pygame.Rect(globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT, globvar.DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS)


test_fonts = ["AndikaNewBasic-B.ttf", "OpenSans-Light.ttf", "Calligraffiti.ttf",
              "Lora-Regular.ttf", "Alef-Regular.ttf", "Lora-Italic.ttf", "Alef-Bold.ttf"]

text_to_lerp = "What does a skeleton tile his roof with?\n" \
               "Shin-gles!"


font1 = test_fonts[4]
font1_bold = test_fonts[6]
font2 = test_fonts[3]
font2_italic = test_fonts[5]
font_sans = "comic_sans.ttf"
font_papyrus = "PAPYRUS.ttf"
font_cursive = "Calligraffiti.ttf"


test_lerped_glyphs = None
test_lerped_glyphs = global_manager.get_glyphs_from_text(text_to_lerp,
                                                         font_sans,
                                                         font_papyrus,
                                                         wrap_x=w)


drawing_full_text_mode = test_lerped_glyphs is not None

# test_o = ttfConverter.glyph_from_font("q", font2)
# test_o.worldspace_scale_by(globvar.EM_TO_FONT_SCALE)
# test_o.update_bounds()
# up_left_origin_diff = test_o.get_upper_left_world()
# test_o.worldspace_offset_by(-up_left_origin_diff)
# test_o.update_bounds()
#
# test_glyphs = []


active_letter_initial_scale = 1
character = "r"
g1_import = ttfConverter.glyph_from_font("j", font1)
g2_import = ttfConverter.glyph_from_font("A", font2)

g1_import.worldspace_scale_by(0.15 * active_letter_initial_scale)
g2_import.worldspace_scale_by(0.15 * active_letter_initial_scale)


global_manager.set_active_glyphs(g1_import, g2_import)

# CIRCLE
# circle = contour.get_unit_circle_contour()
# circle_size = 100
# circle_g = glyph.Glyph()
# circle_g.append_contour(circle)
# circle_g.worldspace_scale_by(circle_size)
# circle_g.worldspace_offset_by(globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT)
# circle_g.worldspace_offset_by(np.array([circle_size, 0], dtype=globvar.POINT_NP_DTYPE))
#
#
#
# # testing quadratic bezier split
# curve1 = circle.curves[0]
# curve_test = glyph.glyph_from_curves([curve1])
#

# polygon_size = 100
# tri = contour.get_unit_polygon_contour(3, polygon_size)
# g_tri = glyph.Glyph()
# g_tri.append_contour(tri)
#
# hex = contour.get_unit_polygon_contour(6, polygon_size, np.pi / 6)
# g_hex = glyph.Glyph()
# g_hex.append_contour(hex)
#
#
#
# global_manager.set_active_glyphs(g_hex, g_tri)


test_points = np.array([[0, 0],
                        [2, 4],
                        [0, 8],
                        [11, -4],
                        [15, -6]])
test_points *= 5



# globvar.test_curves.append()


# Start execution ============================
mixed_glyph = None
mappings = None
global_manager.make_mapping_from_active_glyphs()


# make the active glyphs in nice places
g1, g2 = globvar.active_glyphs
# g1.worldspace_offset_by(np.array([w/4, h/2]))
# g2.worldspace_offset_by(np.array([w*2/4, h/2]))

# Do one step to update offsets and such
for g in globvar.glyphs:
    g.update_bounds()
    g.update_curves_upper_left()
    g.update_all_curve_points()
    g.reset_draw_surface()

scrolling = False
# Execution loop
running = True
while running:

    point_radius = globvar.POINT_DRAW_RADIUS

# Clock Updates
    clock.tick(globvar.FPS)
    globvar.mouse_scroll_directions = globvar.empty_offset
    prev_accuracy = globvar.BEZIER_ACCURACY
    last_scrolling = scrolling

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                globvar.KEY_SPACE_PRESSED = True
                # globvar.DEBUG = not globvar.DEBUG
            if event.key == pygame.K_r:
                raise ValueError("Mapping disabled. Come uncomment this if you want a mapping.")
                # mappings, glyph_score = glyph.find_glyph_null_contour_mapping(test_h, test_i, debug_info=True)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                globvar.KEY_SPACE_PRESSED = False
        elif event.type == pygame.MOUSEWHEEL:
            globvar.mouse_scroll_directions = np.array([event.x, event.y])

# ==== Get Inputs ====

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

    scrolling = not np.all(np.isclose(globvar.mouse_scroll_directions, globvar.empty_offset))
    just_stopped_scrolling = (not scrolling) and last_scrolling

    manager.step()

    panned_camera = cursor.panned_this_frame


    # if there were zoom updates, get update the bezier accuracy and get new t values
    if just_stopped_scrolling:
        global_manager.update_bezier_accuracy()
        if globvar.BEZIER_ACCURACY != prev_accuracy:
            global_manager.calculate_t_array()


    for g in globvar.glyphs:
        glyph_changed = False
        g.update_curves_upper_left()
        # g.update_bounds()

        for cont in g.contours:
            for curve in cont.curves:
                # check for updates in abstract points (whose positions are in WORLDSPACE)
                curve_changed = curve.check_abstract_points()
                glyph_changed = glyph_changed or curve_changed
                # the curve has managed its own worldspace changes;
                # update the curves based on the cursor's panning and offsetting
                if just_stopped_scrolling or panned_camera:     # TODO <<<< some redudancy here; caching the image means we don't need to update the cameraspace points when panning, really...
                    curve.update_points()
                    # print("OH MAN")

        if scrolling or glyph_changed:
            print("GLYPH CHANGED ~LINE 196", g)
            g.reset_draw_surface()




    # for testing worldspace/cameraspace, keep track of a rectangle in space to see how it moves
    glyph_box_offset_corner = globvar.DEFAULT_BOUNDING_BOX_UNIT_UPPER_LEFT - globvar.CAMERA_OFFSET
    glyph_box = pygame.Rect(glyph_box_offset_corner * globvar.CAMERA_ZOOM,
                            globvar.DEFAULT_BOUNDING_BOX_UNIT_DIMENSIONS * globvar.CAMERA_ZOOM)


    # Reset the mixed contour on display
    if globvar.lerped_glyph is not None:
        globvar.lerped_glyph.destroy()
        globvar.lerped_glyph = None


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
    # pygame.draw.rect(screen, colors.BLACK, glyph_box, border_width, corner_rad, corner_rad, corner_rad, corner_rad)

    if globvar.lerped_glyph is not None and globvar.show_mixed_glyph:
        globvar.lerped_glyph.draw(screen, point_radius)
        for active_glyph in globvar.active_glyphs:
            active_glyph.update_all_curve_points()      # TODO in theory can be limited to just when scrolling happens
            active_glyph.draw(screen, point_radius)


    # Some leftover DEBUG code
    # for i, l in enumerate(globvar.marking_test_points_lists):
    #     active_glyph_offset = globvar.active_glyphs[1-i].get_upper_left_camera()
    #     num_points = len(l)
    #     for p_i, p in enumerate(l):
    #         if p_i % 1 == 0:
    #             p = custom_math.world_to_cameraspace(p) + active_glyph_offset
    #             # pygame.draw.circle(screen, (0, 0, 0), custom_math.world_to_cameraspace(p) + active_glyph_offset, (p_i+1) * 5/num_points)
    #             centered_pos = (p[0], p[1])
    #             tsurf, tpos = ptext.draw(str(p_i), color=(0, 0, 0), center=centered_pos)
    #             screen.blit(tsurf, tpos)

    # for i, pnt in enumerate(globvar.random_debug_points):
    #     active_glyph_offset = globvar.active_glyphs[1-i].get_upper_left_camera()
    #     pygame.draw.circle(screen, (0, 255, 0), custom_math.world_to_cameraspace(pnt) + active_glyph_offset, 5)


    if test_lerped_glyphs is not None and globvar.show_lerped_glyph_text:
        global_manager.draw_lerped_text(screen, test_lerped_glyphs)

    if globvar.show_current_glyph_mapping:
        manager.draw_mapping_pillow_projection(screen, mappings)


    for test_curve in globvar.test_curves:
        test_curve.draw(screen, point_radius)


    # GUI drawing
    if globvar.update_screen:
        cursor.draw(screen)
        # toolbar.draw(screen)

    # Debug drawing
    if globvar.DEBUG:
        origin_coords = custom_pygame.np_to_ptext_coords(custom_math.world_to_cameraspace(origin))
        pygame.draw.circle(screen, colors.BLACK, origin_coords, 5)

        tsurf, tpos = ptext.draw(str((0, 0)), color=colors.BLACK,
                                 left=origin_coords[0] + globvar.POINT_DRAW_RADIUS,
                                 top=origin_coords[1] + globvar.POINT_DRAW_RADIUS)
        screen.blit(tsurf, tpos)


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


    pygame.display.flip()




pygame.display.quit()  # should be unnecessary
pygame.quit()
sys.exit()