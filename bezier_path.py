import numpy as np
import pygame
import custom_colors
import global_variables as globvar
import point as AbsPoint

"""
Enables the manipulation of multiple chained Bezier Objects
"""


class BezierPath2D(object):
    def __init__(self):
        self.beziers = []
        self.forms_circuit = False
        return

    def add_bezier(self, bezier):
        self.beziers.append(bezier)
        return



    