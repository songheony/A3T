import sys
import numpy as np
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.samf import SAMF as Tracker


class SAMF(Expert):
    def __init__(self):
        super(SAMF, self).__init__("SAMF")
        self.tracker = Tracker()

    def init(self, image, box):
        image = np.array(image)
        self.tracker.init(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.update(image)
