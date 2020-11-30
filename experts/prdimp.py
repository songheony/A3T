import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pytracking/")
from pytracking.tracker.dimp.dimp import DiMP as Tracker
from pytracking.parameter.dimp.prdimp18 import parameters as prdimp18param
from pytracking.parameter.dimp.prdimp50 import parameters as prdimp50param


class PrDiMP18(BaseTracker):
    def __init__(self):
        super(PrDiMP18, self).__init__("PrDiMP-18")
        self.tracker = Tracker(prdimp18param())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]


class PrDiMP50(BaseTracker):
    def __init__(self):
        super(PrDiMP50, self).__init__("PrDiMP-50")
        self.tracker = Tracker(prdimp50param())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]
