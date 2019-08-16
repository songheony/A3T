import sys
import cv2
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.strcf import STRCF as Tracker
from cftracker.config import strdcf_hc_config


class STRCF(Expert):
    def __init__(self):
        super(STRCF, self).__init__("STRCF")
        self.tracker = Tracker(config=strdcf_hc_config.STRDCFHCConfig())

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox = self.tracker.update(image)
        return bbox
