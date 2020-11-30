import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pytracking/")
from pytracking.tracker.kys.kys import KYS as Tracker
from pytracking.parameter.kys.default import parameters


class KYS(BaseTracker):
    def __init__(self):
        super(KYS, self).__init__("KYS")
        self.tracker = Tracker(parameters())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]
