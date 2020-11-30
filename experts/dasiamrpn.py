import sys
import cv2
import torch
import numpy as np
from base_tracker import BaseTracker
import path_config

sys.path.append("external/DaSiamRPN/code")
from net import SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import cxy_wh_2_rect


class DaSiamRPN(BaseTracker):
    def __init__(self):
        super(DaSiamRPN, self).__init__(name="DaSiamRPN")
        self.net_file = path_config.DASIAMRPN_MODEL

    def initialize(self, image_file, box):
        self.net = SiamRPNotb()
        self.net.load_state_dict(torch.load(self.net_file))
        self.net.eval().cuda()

        image = cv2.imread(image_file)
        box = box - np.array([1, 1, 0, 0])
        self.state = SiamRPN_init(
            image, box[:2] + box[2:] / 2.0, box[2:], self.net
        )  # init tracker

    def track(self, image_file):
        image = cv2.imread(image_file)
        self.state = SiamRPN_track(self.state, image)  # track
        center = self.state["target_pos"] + 1
        target_sz = self.state["target_sz"]
        box = cxy_wh_2_rect(center, target_sz)
        return box
