import sys
import cv2
from .expert import Expert

sys.path.append("external/pyCFTrackers")
from cftracker.samf import SAMF as Tracker


class SAMF(Expert):
    def __init__(self):
        super(SAMF, self).__init__("SAMF")
        self.tracker = Tracker()

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.update(image)
        return bbox
