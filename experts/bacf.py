import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pyCFTrackers")
from cftracker.bacf import BACF as Tracker
from cftracker.config import bacf_config


class BACF(BaseTracker):
    def __init__(self):
        super(BACF, self).__init__("BACF")
        self.tracker = Tracker(config=bacf_config.BACFConfig())

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.prev_box = box
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        self.prev_box = bbox
        return bbox
