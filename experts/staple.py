import sys
import cv2
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.staple import Staple as Tracker
from cftracker.config import staple_config


class Staple(Expert):
    def __init__(self):
        super(Staple, self).__init__("Staple")
        self.tracker = Tracker(config=staple_config.StapleConfig())

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.tracker.update(image)
