import sys
from PIL import Image
from base_tracker import BaseTracker
import path_config

sys.path.append("external/siamrpn-pytorch")
from siamrpn import TrackerSiamRPN


class SiamRPN(BaseTracker):
    def __init__(self):
        super(SiamRPN, self).__init__("SiamRPN")
        self.net_file = path_config.SIAMRPN_MODEL
        self.tracker = TrackerSiamRPN(net_path=self.net_file)

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.tracker.update(image)
