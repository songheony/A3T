import sys
from PIL import Image
from base_tracker import BaseTracker
import path_config

sys.path.append("external/siamfc")
from siamfc import TrackerSiamFC


class SiamFC(BaseTracker):
    def __init__(self):
        super(SiamFC, self).__init__("SiamFC")
        # TODO: edit this path
        self.net_file = path_config.SIAMFC_MODEL
        self.tracker = TrackerSiamFC(net_path=self.net_file)

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.tracker.update(image)
