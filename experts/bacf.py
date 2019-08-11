import sys
import numpy as np
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.bacf import BACF as Tracker


class BACF(Expert):
    def __init__(self):
        super(BACF, self).__init__("BACF")
        self.tracker = Tracker()

    def init(self, image, box):
        image = np.array(image)
        self.tracker.init(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.update(image)
