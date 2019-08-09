from .expert import Expert
from external.siamrpn_pytorch.siamrpn import TrackerSiamRPN


class SiamRPN(Expert):
    def __init__(self):
        super(SiamRPN, self).__init__("SiamRPN")
        # TODO: edit this path
        self.net_file = "/home/heonsong/Desktop/AAA/AAA-journal/external/siamrpn_pytorch/model.pth"

    def init(self, image, box):
        self.tracker = TrackerSiamRPN(net_path=self.net_file)
        self.tracker.init(image, box)

    def update(self, image):
        return self.tracker.update(image)
