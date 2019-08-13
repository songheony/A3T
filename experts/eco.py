import sys
import cv2
from .expert import Expert

sys.path.append("external/pyECO")
from eco import ECOTracker


class ECO(Expert):
    def __init__(self):
        super(ECO, self).__init__("ECO")

    def initialize(self, image, box):
        self.tracker = ECOTracker(True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.tracker.update(image, True, False)
