import sys
import cv2
import torch
import numpy as np
from base_tracker import BaseTracker

sys.path.append("external/SPM-Tracker/")
from siam_tracker.core.inference import SiamTracker
from siam_tracker.core.config import merge_cfg_from_file


def img2tensor(img, device):
    """ Convert numpy.ndarry to torch.Tensor """
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().to(device)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


class SPM(BaseTracker):
    def __init__(self):
        super(SPM, self).__init__("SPM")
        cfg_path = "/home/heonsong/Desktop/AAA/AAA-journal/external/SPM-Tracker/configs/spm_tracker/alexnet_c42_otb.yaml"
        gpu_id = 0
        merge_cfg_from_file(cfg_path)
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = SiamTracker(self.device)

    def initialize(self, image_file, box):
        frame = cv2.imread(image_file)
        img_tensor = img2tensor(frame, self.device)
        box = [(box[0]), (box[1]), (box[2] + box[0]), (box[3] + box[1])]
        self.model.tracker.init_tracker(img_tensor, box)
        self.current_box = box

    def track(self, image_file):
        frame = cv2.imread(image_file)
        img_tensor = img2tensor(frame, self.device)
        self.current_box = self.model.tracker.predict_next_frame(
            img_tensor, self.current_box
        )
        bbox = np.array(self.current_box[:])
        bbox[2:] -= bbox[:2]
        return bbox
