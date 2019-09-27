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

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        return bbox
