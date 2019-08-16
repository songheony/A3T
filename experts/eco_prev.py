import sys
import numpy as np
from .expert import Expert

sys.path.append("external/pyECO")
from eco import ECOTracker


class ECO_prev(Expert):
    def __init__(self):
        super(ECO_prev, self).__init__("ECO_prev")

    def initialize(self, image, box):
        if np.all(image[:, :, 0] == image[:, :, 1]):
            self.tracker = ECOTracker(is_color=False)
            image = image[:, :, :1]
        else:
            self.tracker = ECOTracker(is_color=True)
        self.tracker.init(image, box)

    def track(self, image):
        if not self.tracker._is_color:
            image = image[:, :, :1]

        bbox = self.tracker.update(image, train=True, vis=False)
        x1, y1, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = np.array([x1, y1, w, h])
        return pos
