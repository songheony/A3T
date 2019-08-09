import numpy as np
import sys
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.atom.atom import ATOM as Tracker
from pytracking.parameter.atom import default


class ATOM(Expert):
    def __init__(self):
        super(ATOM, self).__init__("ATOM")
        self.tracker = Tracker(default.parameters())

    def init(self, image, box):
        image = np.array(image)
        self.tracker.initialize(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.track(image)
