import sys
import cv2
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.samf import SAMF as Tracker


class SAMF(Expert):
    def __init__(self):
        super(SAMF, self).__init__("SAMF")
        self.tracker = Tracker()

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox = self.tracker.update(image)
        return bbox
