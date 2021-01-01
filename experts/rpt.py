import sys
import cv2
import numpy as np
from base_tracker import BaseTracker
import path_config

sys.path.append("external/RPT")
from siamreppoints.core.config import cfg
from siamreppoints.models.model_builder import ModelBuilder
from siamreppoints.tracker.tracker_builder import build_tracker
from siamreppoints.utils.model_load import load_pretrain
from siamreppoints.utils.bbox import get_axis_aligned_bbox


class RPT(BaseTracker):
    def __init__(self):
        super(RPT, self).__init__("RPT")

        # load config
        cfg.merge_from_file(path_config.RPT_CONFIG)

        # create model
        model = ModelBuilder()

        # load model
        model = load_pretrain(model, path_config.RPT_SNAPSHOT).cuda().eval()

        # build tracker
        self.tracker = build_tracker(model)

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        cx, cy, w, h = get_axis_aligned_bbox(np.array(box))
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        self.tracker.init(image, gt_bbox_)

    def track(self, image_file):
        image = cv2.imread(image_file)
        outputs = self.tracker.track(image)
        pred_bbox = outputs["bbox"]
        return pred_bbox
