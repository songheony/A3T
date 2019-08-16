import sys
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.eco.eco import ECO as Tracker
from pytracking.parameter.eco import default


class ECO_new(Expert):
    def __init__(self):
        super(ECO_new, self).__init__("ECO_new")
        self.tracker = Tracker(default.parameters())

    def initialize(self, image, box):
        self.tracker.initialize(image, box)

    def track(self, image):
        return self.tracker.track(image)
