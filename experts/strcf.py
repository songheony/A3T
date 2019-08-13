import sys
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.strcf import STRCF as Tracker


class STRCF(Expert):
    def __init__(self):
        super(STRCF, self).__init__("STRCF")
        self.tracker = Tracker()

    def initialize(self, image, box):
        self.tracker.init(image, box)

    def track(self, image):
        return self.tracker.update(image)
