import sys
import cv2
import torch
import numpy as np
from .expert import Expert

sys.path.append("external/DaSiamRPN/code")
from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import cxy_wh_2_rect


class DaSiamRPN(Expert):
    def __init__(self):
        super(DaSiamRPN, self).__init__(name="DaSiamRPN")
        self.net_file = (
            "/home/heonsong/Desktop/AAA/AAA-journal/external/DaSiamRPN/SiamRPNBIG.model"
        )

    def init(self, image, box):
        self.net = SiamRPNBIG()
        self.net.load_state_dict(torch.load(self.net_file))
        self.net.eval().cuda()

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        box = box - [1, 1, 0, 0]
        self.state = SiamRPN_init(
            image, box[:2] + box[2:] / 2.0, box[2:], self.net
        )  # init tracker

    def update(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = SiamRPN_track(self.state, image)  # track
        center = self.state["target_pos"] + 1
        target_sz = self.state["target_sz"]
        box = cxy_wh_2_rect(center, target_sz)
        return box
