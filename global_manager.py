import global_variables as globvar
import numpy as np

class GlobalManager(object):
    def __init__(self):
        return

    """
    Based on the level of zoom, adjust the accuracy with which Bezier curves are drawn.
    """
    def update_bezier_accuracy(self):
        globvar.BEZIER_ACCURACY = int(2 * np.log2(globvar.CAMERA_ZOOM * 30)) + 1
        return

    def calculate_t_array(self):
        globvar.t_values = np.array([[(i / globvar.BEZIER_ACCURACY) ** 3,
                                      (i / globvar.BEZIER_ACCURACY) ** 2,
                                      (i / globvar.BEZIER_ACCURACY), 1] for i in range(globvar.BEZIER_ACCURACY + 1)],
                                    dtype=globvar.POINT_NP_DTYPE)
        return