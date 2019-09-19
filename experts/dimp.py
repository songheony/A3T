import sys
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.dimp.dimp import DiMP as Tracker
from pytracking.parameter.dimp import dimp50


class DiMP(Expert):
    def __init__(self):
        super(DiMP, self).__init__("DiMP")
        self.tracker = Tracker(dimp50.parameters())

    def initialize(self, image, box):
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image):
        return self.tracker.track(image)["target_bbox"]