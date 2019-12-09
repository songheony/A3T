import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pyCFTrackers")
from cftracker.csrdcf import CSRDCF as Tracker
from cftracker.config import csrdcf_config


class CSRDCF(BaseTracker):
    def __init__(self):
        super(CSRDCF, self).__init__("CSRDCF")
        self.tracker = Tracker(config=csrdcf_config.CSRDCFConfig())

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        return bbox
