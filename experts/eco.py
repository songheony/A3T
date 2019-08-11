import sys
import numpy as np
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.eco import ECO as Tracker
from lib.eco.config import otb_deep_config


class ECO(Expert):
    def __init__(self):
        super(ECO, self).__init__("ECO")
        self.tracker = Tracker(config=otb_deep_config.OTBDeepConfig())

    def init(self, image, box):
        image = np.array(image)
        self.tracker.init(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.update(image)
