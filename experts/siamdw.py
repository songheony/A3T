import numpy as np
import cv2
from easydict import EasyDict as edict
from .expert import Expert
from external.SiamDW.lib.tracker.siamfc import SiamFC
import external.SiamDW.lib.models.models as models
from external.SiamDW.lib.utils.utils import load_pretrain


class SiamDW(Expert):
    def __init__(self):
        super().__init__("SiamDW")
        # TODO: edit this path
        net_file = "/home/heonsong/Desktop/AAA/AAA-journal/external/SiamDW/CIResNet22FC_G.pth"
        info = edict()
        info.arch = "SiamFCRes22"
        info.dataset = "OTB"
        info.epoch_test = False
        self.tracker = SiamFC(info)
        self.net = models.__dict__["SiamFCRes22"](anchors_nums=5)
        self.net = load_pretrain(self.net, net_file)
        self.net.eval()
        self.net = self.net.cuda()

    def init(self, image, box):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        center = np.array([box[0] + (box[2] - 1) / 2, box[1] + (box[3] - 1) / 2])
        size = np.array([box[2], box[3]])
        self.state = self.tracker.init(image, center, size, self.net)

    def update(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = self.tracker.track(self.state, image)
        center = self.state["target_pos"]
        size = self.state["target_sz"]
        bbox = center[0] - size[0] / 2, center[1] - size[1] / 2, size[0], size[1]
        return bbox
