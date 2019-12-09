import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pyCFTrackers")
from cftracker.staple import Staple as Tracker
from cftracker.config import staple_config


class Staple(BaseTracker):
    def __init__(self):
        super(Staple, self).__init__("Staple")
        self.tracker = Tracker(config=staple_config.StapleConfig())

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        return bbox
