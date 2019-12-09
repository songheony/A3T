import sys
from PIL import Image
from base_tracker import BaseTracker

sys.path.append("external/siamrpn_pytorch")
from siamrpn import TrackerSiamRPN


class SiamRPN(BaseTracker):
    def __init__(self):
        super(SiamRPN, self).__init__("SiamRPN")
        # TODO: edit this path
        self.net_file = (
            "/home/heonsong/Desktop/AAA/AAA-journal/external/siamrpn_pytorch/model.pth"
        )

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker = TrackerSiamRPN(net_path=self.net_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.tracker.update(image)
