import sys
import cv2
import numpy as np
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.bacf import BACF as Tracker
from cftracker.config import bacf_config


class BACF(Expert):
    def __init__(self):
        super(BACF, self).__init__("BACF")
        self.tracker = Tracker(config=bacf_config.BACFConfig())

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox = self.tracker.update(image)
        bbox = np.array(bbox, dtype=int)
        return bbox
