import sys
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.csrdcf import CSRDCF as Tracker
from cftracker.config import csrdcf_config


class CSRDCF(Expert):
    def __init__(self):
        super(CSRDCF, self).__init__("CSRDCF")
        self.tracker = Tracker(config=csrdcf_config.CSRDCFConfig())

    def initialize(self, image, box):
        self.tracker.init(image, box)

    def track(self, image):
        return self.tracker.update(image)
