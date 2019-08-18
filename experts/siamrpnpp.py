import sys
import cv2
import torch
from .expert import Expert

sys.path.append("external/pysot")
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


class SiamRPNPP(Expert):
    def __init__(self):
        super(SiamRPNPP, self).__init__("SiamRPN++")
        # TODO: edit this path
        config = "/home/heonsong/Desktop/AAA/AAA-journal/external/pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml"
        snapshot = "/home/heonsong/Desktop/AAA/AAA-journal/external/pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth"

        # load config
        cfg.merge_from_file(config)
        cfg.CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if cfg.CUDA else "cpu")

        # create model
        self.model = ModelBuilder()

        # load model
        self.model.load_state_dict(
            torch.load(snapshot, map_location=lambda storage, loc: storage.cpu())
        )
        self.model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(self.model)

    def initialize(self, image, box):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.tracker.init(image, box)

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox = self.tracker.track(image)['bbox']
        return bbox
