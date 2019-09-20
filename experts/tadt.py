import sys
import numpy as np
import cv2
import torch
from .expert import Expert

sys.path.append("external/TADT_python")
from tadt_tracker import Tadt_Tracker
from defaults import _C as cfg
from backbone_v2 import build_vgg16


class TADT(Expert):
    def __init__(self):
        super(TADT, self).__init__("TADT")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_vgg16(cfg)

    def initialize(self, image, box):
        self.idx = 0
        self.tracker = Tadt_Tracker(
            cfg, model=self.model, device=self.device, display=False
        )
        self.tracker.initialize_tadt(image, box)

    def track(self, image):
        self.idx += 1
        self.tracker.tracking(image, self.idx)
        return self.tracker.results[-1]
