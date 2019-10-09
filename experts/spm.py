import sys
import cv2
import torch
from .expert import Expert

sys.path.append("external/SPM-Tracker/")
from siam_tracker.core.inference import SiamTracker
from siam_tracker.core.config import merge_cfg_from_file


def img2tensor(img, device):
    """ Convert numpy.ndarry to torch.Tensor """
    img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float().to(device)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


class SPM(Expert):
    def __init__(self):
        super(SPM, self).__init__("SPM")
        cfg_path = ""
        gpu_id = 0
        merge_cfg_from_file(cfg_path)
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = SiamTracker(self.device)

    def initialize(self, image_file, box):
        frame = cv2.imread(image_file)
        img_tensor = img2tensor(frame, self.device)
        self.model.tracker.init_tracker(img_tensor, box)
        self.current_box = box

    def track(self, image_file):
        frame = cv2.imread(image_file)
        img_tensor = img2tensor(frame, self.device)
        self.current_box = self.model.tracker.predict_next_frame(
            img_tensor, self.current_box
        )
        return self.current_box
