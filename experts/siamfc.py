import sys
from .expert import Expert

sys.path.append("external/siamfc_pytorch")
from siamfc.siamfc import TrackerSiamFC


class SiamFC(Expert):
    def __init__(self):
        super(SiamFC, self).__init__("SiamFC")
        # TODO: edit this path
        self.net_file = (
            "/home/heonsong/Desktop/AAA/AAA-journal/external/siamfc_pytorch/model.pth"
        )

    def init(self, image, box):
        self.tracker = TrackerSiamFC(net_path=self.net_file)
        self.tracker.init(image, box)

    def update(self, image):
        return self.tracker.update(image)
