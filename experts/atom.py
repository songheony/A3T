import numpy as np
from .expert import Expert
from external.pytracking.pytracking.tracker.atom.atom import ATOM as Tracker


class ATOM(Expert):
    def __init__(self):
        super(ATOM, self).__init__("ATOM")
        self.tracker = Tracker()

    def init(self, image, box):
        image = np.array(image)
        self.tracker.initialize(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.track(image)
