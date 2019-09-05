import sys
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.eco.eco import ECO as Tracker
from pytracking.parameter.eco import default


class ECO(Expert):
    def __init__(self):
        super(ECO, self).__init__("ECO")
        self.tracker = Tracker(default.parameters())

    def initialize(self, image, box):
        state = {'init_bbox': box}
        self.tracker.initialize(image, state)

    def track(self, image):
        return self.tracker.track(image)['target_bbox']
