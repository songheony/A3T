import sys
import cv2
import torch
import numpy as np
from base_tracker import BaseTracker
import path_config

sys.path.append("external/DROL")
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain


class DROL(BaseTracker):
    def __init__(self):
        super(DROL, self).__init__("DROL")

        # load config
        cfg.merge_from_file(path_config.DROL_CONFIG)

        # create model
        model = ModelBuilder()

        # load model
        model = load_pretrain(model, path_config.DROL_SNAPSHOT).cuda().eval()

        # build tracker
        self.tracker = build_tracker(model)

        # warmup
        for i in range(10):
            model.template(torch.FloatTensor(1,3,127,127).cuda())

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
