import global_variables as globvar

class Glyph(object):
    def __init__(self):
        globvar.glyphs.append(self)

        self.contours = []

        self.origin_offset = globvar.empty_offset.copy()

        return

    def append_contour(self, contour):
        self.contours.append(contour)
        return

    def draw(self, surface, radius, color_gradient=True):
        for contour in self.contours:
            contour.draw(surface, radius, color_gradient)
        return

def find_glyph_contour_mapping(glyph1, glyph2):
    return

def lerp_glyphs(glyph1, glyph2, lerp_value):
    return