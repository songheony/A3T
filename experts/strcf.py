import sys
import numpy as np
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.strcf import STRCF as Tracker


class STRCF(Expert):
    def __init__(self):
        super(STRCF, self).__init__("STRCF")
        self.tracker = Tracker()

    def init(self, image, box):
        image = np.array(image)
        self.tracker.init(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.update(image)
