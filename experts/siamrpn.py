import sys
from PIL import Image
from .expert import Expert

sys.path.append("external/siamrpn_pytorch")
from siamrpn import TrackerSiamRPN


class SiamRPN(Expert):
    def __init__(self):
        super(SiamRPN, self).__init__("SiamRPN")
        # TODO: edit this path
        self.net_file = (
            "/home/heonsong/Desktop/AAA/AAA-journal/external/siamrpn_pytorch/model.pth"
        )

    def initialize(self, image, box):
        image = Image.fromarray(image)
        self.tracker = TrackerSiamRPN(net_path=self.net_file)
        self.tracker.init(image, box)

    def track(self, image):
        image = Image.fromarray(image)
        return self.tracker.update(image)
