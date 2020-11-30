import sys
import cv2
from base_tracker import BaseTracker

sys.path.append("external/pytracking/")
from pytracking.tracker.dimp.dimp import DiMP as Tracker
from pytracking.parameter.dimp.dimp18 import parameters as dimp18param
from pytracking.parameter.dimp.dimp50 import parameters as dimp50param


class DiMP18(BaseTracker):
    def __init__(self):
        super(DiMP18, self).__init__("DiMP-18")
        self.tracker = Tracker(dimp18param())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]


class DiMP50(BaseTracker):
    def __init__(self):
        super(DiMP50, self).__init__("DiMP-50")
        self.tracker = Tracker(dimp50param())

    def initialize(self, image_file, box):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        state = {"init_bbox": box}
        self.tracker.initialize(image, state)

    def track(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        return self.tracker.track(image)["target_bbox"]
