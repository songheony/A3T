import sys
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.dimp.dimp import DiMP as Tracker
from pytracking.parameter.dimp import dimp18


class DiMP18(Expert):
    def __init__(self):
        super(DiMP18, self).__init__("DiMP18")
        self.tracker = Tracker(dimp18.parameters())

    def initialize(self, image, box):
        self.tracker.initialize(image, box)

    def track(self, image):
        return self.tracker.track(image)
