import sys
import cv2
from .expert import Expert

sys.path.append("external/pytracking/")
from pytracking.tracker.eco.eco import ECO as Tracker
from pytracking.parameter.eco import default


class ECO(Expert):
    def __init__(self):
        super(ECO, self).__init__("ECO")
        self.tracker = Tracker(default.parameters())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]
