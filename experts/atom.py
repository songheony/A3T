import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pytracking/")
from pytracking.tracker.atom.atom import ATOM as Tracker
from pytracking.parameter.atom import default


class ATOM(BaseTracker):
    def __init__(self):
        super(ATOM, self).__init__("ATOM")
        self.tracker = Tracker(default.parameters())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]
