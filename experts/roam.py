import os
import sys
from PIL import Image
import numpy as np
import path_config
from base_tracker import BaseTracker

sys.path.append("external/ROAM")
from utils import list_models
from networks import FeatureExtractor
from tracker import Tracker


class ROAM(BaseTracker):
    def __init__(self):
        super(ROAM, self).__init__("ROAM")
        feat_extractor = FeatureExtractor(path_config.ROAM_FEAT_DIR)
        self.tracker = Tracker(feat_extractor, is_debug=False)
        models = list_models(os.path.abspath(path_config.ROAM_MODEL_DIR))
        self.tracker.load_models(models[-1])

    def initialize(self, image_file, box):
        img = np.array(Image.open(image_file).convert("RGB"))
        self.tracker.initialize(img, box)

    def track(self, image_file):
        img = np.array(Image.open(image_file).convert("RGB"))
        return self.tracker.track(img)
