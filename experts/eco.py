import numpy as np
from .expert import Expert
from external.pyCFTrackers.cftracker.eco import ECO as Tracker
from external.pyCFTrackers.lib.eco.config import otb_deep_config


class ECO(Expert):
    def __init__(self):
        super(ECO, self).__init__("ECO")
        self.tracker = Tracker(config=otb_deep_config.OTBDeepConfig())

    def init(self, image, box):
        image = np.array(image)
        self.tracker.initialize(image, box)

    def update(self, image):
        image = np.array(image)
        return self.tracker.track(image)
