import sys
import numpy as np
import torch
from .expert import Expert

sys.path.append("external/UDT_pytorch/")
from track.UDT import TrackerConfig
from track.util import crop_chw, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from track.net import DCFNet


class UDT(Expert):
    def __init__(self):
        super(UDT, self).__init__("UDT")
        self.model = "/home/heonsong/Desktop/AAA/AAA-journal/external/UDT_pytorch/train/work/checkpoint.pth.tar"
        self.net = DCFNet(self.config)

    def initialize(self, image, box):
        # default parameter and load feature extractor network
        self.config = TrackerConfig()
        self.net.load_param(self.model)
        self.net.eval().cuda()

        self.target_pos, self.target_sz = rect1_2_cxy_wh(box)

        # confine results
        self.min_sz = np.maximum(self.config.min_scale_factor * self.target_sz, 4)
        self.max_sz = np.minimum(
            image.shape[:2], self.config.max_scale_factor * self.target_sz
        )

        # crop template
        self.window_sz = self.target_sz * (1 + self.config.padding)
        self.bbox = cxy_wh_2_bbox(self.target_pos, self.window_sz)
        self.patch = crop_chw(image, self.bbox, self.config.crop_sz)

        self.target = self.patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(self.target, axis=0)).cuda())

        self.res = [cxy_wh_2_rect1(self.target_pos, self.target_sz)]  # save in .txt
        self.patch_crop = np.zeros(
            (
                self.config.num_scale,
                self.patch.shape[0],
                self.patch.shape[1],
                self.patch.shape[2],
            ),
            np.float32,
        )

    def track(self, image):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            self.window_sz = self.target_sz * (
                self.config.scale_factor[i] * (1 + self.config.padding)
            )
            self.bbox = cxy_wh_2_bbox(self.target_pos, self.window_sz)
            self.patch_crop[i, :] = crop_chw(image, self.bbox, self.config.crop_sz)

        self.search = self.patch_crop - self.config.net_average_image
        self.response = self.net(torch.Tensor(self.search).cuda())
        self.peak, idx = torch.max(self.response.view(self.config.num_scale, -1), 1)
        self.peak = self.peak.data.cpu().numpy() * self.config.scale_penalties
        self.best_scale = np.argmax(self.peak)
        self.r_max, self.c_max = np.unravel_index(
            idx[self.best_scale], self.config.net_input_size
        )

        if self.r_max > self.config.net_input_size[0] / 2:
            self.r_max = self.r_max - self.config.net_input_size[0]
        if self.c_max > self.config.net_input_size[1] / 2:
            self.c_max = self.c_max - self.config.net_input_size[1]
        self.window_sz = self.target_sz * (
            self.config.scale_factor[self.best_scale] * (1 + self.config.padding)
        )

        self.target_pos = (
            self.target_pos
            + np.array([self.c_max, self.r_max])
            * self.window_sz
            / self.config.net_input_size
        )
        self.target_sz = np.minimum(
            np.maximum(self.window_sz / (1 + self.config.padding), self.min_sz),
            self.max_sz,
        )

        # model update
        self.window_sz = self.target_sz * (1 + self.config.padding)
        self.bbox = cxy_wh_2_bbox(self.target_pos, self.window_sz)
        self.patch = crop_chw(image, self.bbox, self.config.crop_sz)
        self.target = self.patch - self.config.net_average_image
        self.net.update(
            torch.Tensor(np.expand_dims(self.target, axis=0)).cuda(),
            lr=self.config.interp_factor,
        )

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)
