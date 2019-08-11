import sys
import numpy as np
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.staple import Staple as Tracker
from lib.eco.config import staple_config


class Staple(Expert):
    def __init__(self):
        super(Staple, self).__init__("Staple")
        self.tracker = Tracker(config=staple_config.StapleConfig())

    def init(self, image, box):
        image = np.array(image)
        self.tracker.init(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.update(image)
