import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pyCFTrackers")
from cftracker.ldes import LDES as Tracker
from cftracker.config import ldes_config


class LDES(BaseTracker):
    def __init__(self):
        super(LDES, self).__init__("LDES")
        self.tracker = Tracker(config=ldes_config.LDESDemoLinearConfig())

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.prev_box = box
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        self.prev_box = bbox
        return bbox
