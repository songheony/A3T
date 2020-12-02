import sys
from tqdm import tqdm
import numpy as np
import torch
import cv2
from easydict import EasyDict as edict
from base_tracker import BaseTracker
import path_config

sys.path.append("external/TracKit/lib")
from tracker.ocean import Ocean as Tracker
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox
import models.models as models


class Ocean(BaseTracker):
    def __init__(self):
        super(Ocean, self).__init__("Ocean")

        siam_info = edict()
        siam_info.arch = "Ocean"
        siam_info.dataset = "OTB2015"
        siam_info.online = False
        siam_info.epoch_test = False
        siam_info.TRT = False
        siam_info.align = False

        self.siam_tracker = Tracker(siam_info)
        self.siam_net = models.__dict__["Ocean"](align=siam_info.align, online=False)
        self.siam_net = load_pretrain(self.siam_net, path_config.OCEAN_MODEL)
        self.siam_net.eval()
        self.siam_net = self.siam_net.cuda()

        # warmup
        for i in tqdm(range(100)):
            self.siam_net.template(torch.rand(1, 3, 127, 127).cuda())
            self.siam_net.track(torch.rand(1, 3, 255, 255).cuda())

    def initialize(self, image_file, box):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training

        cx, cy, w, h = get_axis_aligned_bbox(box)
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])

        self.state = self.siam_tracker.init(
            im, target_pos, target_sz, self.siam_net
        )  # init tracker

    def track(self, image_file):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training

        self.state = self.siam_tracker.track(self.state, im)
        location = cxy_wh_2_rect(self.state["target_pos"], self.state["target_sz"])
        return location
