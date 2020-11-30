import sys
import numpy as np
import cv2
from easydict import EasyDict as edict
from base_tracker import BaseTracker
import path_config

sys.path.append("external/SiamDW/lib")
from tracker.siamfc import SiamFC
import models.models as models
from utils.utils import load_pretrain


class SiamDW(BaseTracker):
    def __init__(self):
        super().__init__("SiamDW_SiamFCRes22W_G")
        net_file = path_config.SIAMDW_SIAMFCRES22W_MODEL
        info = edict()
        info.arch = "SiamFCRes22"
        info.dataset = "OTB2015"
        info.epoch_test = False
        self.tracker = SiamFC(info)
        self.net = models.__dict__["SiamFCRes22W"]()
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
