import sys
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.samf import SAMF as Tracker


class SAMF(Expert):
    def __init__(self):
        super(SAMF, self).__init__("SAMF")
        self.tracker = Tracker()

    def initialize(self, image, box):
        self.tracker.init(image, box)

    def track(self, image):
        return self.tracker.update(image)
