import sys
import cv2
from .expert import Expert

import faulthandler

sys.path.append("external/pyCFTrackers")
from cftracker.eco import ECO as Tracker
from lib.eco.config import otb_hc_config


class ECO_HC(Expert):
    def __init__(self):
        super(ECO_HC, self).__init__("ECO-HC")
        self.tracker = Tracker(config=otb_hc_config.OTBHCConfig())
        faulthandler.enable()

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)

        return bbox
