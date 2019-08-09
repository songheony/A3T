from .expert import Expert
from external.siamfc_pytorch.siamfc.siamfc import TrackerSiamFC


class SiamFC(Expert):
    def __init__(self):
        super(SiamFC, self).__init__("SiamFC")
        self.net_file = "pretrained/siamfc_alexnet_e50.pth"

    def init(self, image, box):
        self.tracker = TrackerSiamFC(net_path=self.net_file)
        self.tracker.init(image, box)

    def update(self, image):
        return self.tracker.update(image)
