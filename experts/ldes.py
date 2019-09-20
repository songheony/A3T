import sys
import cv2
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.ldes import LDES as Tracker
from cftracker.config import ldes_config


class LDES(Expert):
    def __init__(self):
        super(LDES, self).__init__("LDES")
        self.tracker = Tracker(config=ldes_config.LDESDemoLinearConfig())

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox = self.tracker.update(image)
        return bbox
