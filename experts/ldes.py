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

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        return bbox
