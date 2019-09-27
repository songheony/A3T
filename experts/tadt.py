import sys
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

    def initialize(self, image_file, box):
        self.idx = 0
        self.tracker = Tadt_Tracker(
            cfg, model=self.model, device=self.device, display=False
        )
        self.tracker.initialize_tadt(image_file, box)

    def track(self, image_file):
        self.idx += 1
        self.tracker.tracking(image_file, self.idx)
        return self.tracker.results[-1]
