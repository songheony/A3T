import sys
import cv2
import numpy as np
from base_tracker import BaseTracker
import path_config

sys.path.append("external/SiamCAR/")
from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder


class SiamCAR(BaseTracker):
    def __init__(self):
        super(SiamCAR, self).__init__("SiamCAR")

        # load config
        cfg.merge_from_file(path_config.SIAMCAR_CONFIG)

        # hp_search
        params = getattr(cfg.HP_SEARCH, "OTB100")
        self.hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

        model = ModelBuilder()

        # load model
        model = load_pretrain(model, path_config.SIAMCAR_SNAPSHOT).cuda().eval()

        # build tracker
        self.tracker = SiamCARTracker(model, cfg.TRACK)

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        cx, cy, w, h = get_axis_aligned_bbox(np.array(box))
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        self.tracker.init(image, gt_bbox_)

    def track(self, image_file):
        image = cv2.imread(image_file)
        outputs = self.tracker.track(image, self.hp)
        pred_bbox = outputs["bbox"]
        return pred_bbox
