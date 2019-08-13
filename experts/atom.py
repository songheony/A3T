import sys
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.atom.atom import ATOM as Tracker
from pytracking.parameter.atom import default


class ATOM(Expert):
    def __init__(self):
        super(ATOM, self).__init__("ATOM")
        self.tracker = Tracker(default.parameters())

    def initialize(self, image, box):
        self.tracker.initialize(image, box)

    def track(self, image):
        return self.tracker.track(image)
