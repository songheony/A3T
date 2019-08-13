import sys
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.bacf import BACF as Tracker


class BACF(Expert):
    def __init__(self):
        super(BACF, self).__init__("BACF")
        self.tracker = Tracker()

    def initialize(self, image, box):
        self.tracker.init(image, box)

    def track(self, image):
        return self.tracker.update(image)
