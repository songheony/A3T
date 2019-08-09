import numpy as np
import cv2
import torch
from .expert import Expert
from external.UDT_pytorch.track.UDT import TrackerConfig
from external.UDT_pytorch.track.util import (
    crop_chw,
    cxy_wh_2_rect1,
    rect1_2_cxy_wh,
    cxy_wh_2_bbox,
)
from external.UDT_pytorch.track.net import DCFNet


class UDT(Expert):
    def __init__(self):
        super(UDT, self).__init__("UDT")
        # default parameter and load feature extractor network
        self.net_file = ""
        self.config = TrackerConfig()
        self.net = DCFNet(self.config)
        self.net.load_param(self.net_file)
        self.net.eval().cuda()

    def init(self, image, box):
        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        self.target_pos, self.target_sz = rect1_2_cxy_wh(box)

        # confine results
        self.min_sz = np.maximum(self.config.min_scale_factor * self.target_sz, 4)
        self.max_sz = np.minimum(
            im.shape[:2], self.config.max_scale_factor * self.target_sz
        )

        # crop template
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)

        target = patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

        self.res = [cxy_wh_2_rect1(self.target_pos, self.target_sz)]  # save in .txt
        self.patch_crop = np.zeros(
            (self.config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]),
            np.float32,
        )

    def update(self, image):
        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (
                self.config.scale_factor[i] * (1 + self.config.padding)
            )
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = self.patch_crop - self.config.net_average_image
        response = self.net(torch.Tensor(search).cuda())
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (
            self.config.scale_factor[best_scale] * (1 + self.config.padding)
        )

        self.target_pos = (
            self.target_pos
            + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        )
        self.target_sz = np.minimum(
            np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz
        )

        # model update
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, self.window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)
        target = patch - self.config.net_average_image
        self.net.update(
            torch.Tensor(np.expand_dims(target, axis=0)).cuda(),
            lr=self.config.interp_factor,
        )

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index
