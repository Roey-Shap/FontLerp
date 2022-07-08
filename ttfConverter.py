"""
https://stackoverflow.com/questions/40437308/retrieving-bounding-box-and-bezier-curve-data-for-all-glyphs-in-a-ttf-font-file
"""

import ttfquery
import ttfquery.glyph as glyph

char = "a"
font_url = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
font = ttfquery.describe.openFont(font_url)
g = glyph.Glyph(ttfquery.glyphquery.glyphName(f, char))
contours = g.calculateContours(font)
for contour in contours:
    for point, flag in contour:
        print(point, flag)
