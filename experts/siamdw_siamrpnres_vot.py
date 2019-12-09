import sys
import numpy as np
import cv2
from easydict import EasyDict as edict
from base_tracker import BaseTracker

sys.path.append("external/SiamDW/lib")
from tracker.siamrpn import SiamRPN
import models.models as models
from utils.utils import load_pretrain


class SiamDW(BaseTracker):
    def __init__(self):
        super().__init__("SiamDW_SiamRPNRes22_VOT")
        # TODO: edit this path
        net_file = (
            "/home/heonsong/Desktop/AAA/AAA-weights/external/SiamDW/CIResNet22_RPN.pth"
        )
        info = edict()
        info.arch = "SiamRPNRes22"
        info.dataset = "VOT2017"
        info.epoch_test = False
        info.cls_type = "thinner"
        self.tracker = SiamRPN(info)
        self.net = models.__dict__[info.arch](anchors_nums=5, cls_type=info.cls_type)
        self.net = load_pretrain(self.net, net_file)
        self.net.eval()
        self.net = self.net.cuda()

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        center = np.array([box[0] + (box[2] - 1) / 2, box[1] + (box[3] - 1) / 2])
        size = np.array([box[2], box[3]])
        self.state = self.tracker.init(image, center, size, self.net)

    def track(self, image_file):
        image = cv2.imread(image_file)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.state = self.tracker.track(self.state, image)
        center = self.state["target_pos"]
        size = self.state["target_sz"]
        bbox = (center[0] - size[0] / 2, center[1] - size[1] / 2, size[0], size[1])
        return bbox
