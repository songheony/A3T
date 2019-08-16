import sys
import cv2
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.csrdcf import CSRDCF as Tracker
from cftracker.config import csrdcf_config


class CSRDCF(Expert):
    def __init__(self):
        super(CSRDCF, self).__init__("CSRDCF")
        self.tracker = Tracker(config=csrdcf_config.CSRDCFConfig())

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox = self.tracker.update(image)
        return bbox
