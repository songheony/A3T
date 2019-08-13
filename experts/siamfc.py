import sys
from PIL import Image
from .expert import Expert

sys.path.append("external/siamfc")
from siamfc import TrackerSiamFC


class SiamFC(Expert):
    def __init__(self):
        super(SiamFC, self).__init__("SiamFC")
        # TODO: edit this path
        self.net_file = (
            "/home/heonsong/Desktop/AAA/AAA-journal/external/siamfc/model.pth"
        )

    def initialize(self, image, box):
        image = Image.fromarray(image)
        self.tracker = TrackerSiamFC(net_path=self.net_file)
        self.tracker.init(image, box)

    def track(self, image):
        image = Image.fromarray(image)
        return self.tracker.update(image)
