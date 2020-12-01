import sys
import cv2
import torch
import path_config
from base_tracker import BaseTracker

sys.path.append("external/video_analyst")
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder


class SiamFCPP(BaseTracker):
    def __init__(self):
        super(SiamFCPP, self).__init__("SiamFC++")

        root_cfg.merge_from_file(path_config.SIAMFCPP_CONFIG)

        task = "track"
        task_cfg = root_cfg["test"][task]
        task_cfg.freeze()

        # build model
        model = model_builder.build(task, task_cfg.model)
        # build pipeline
        self.pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)
        dev = torch.device("cuda")
        self.pipeline.set_device(dev)

    def initialize(self, image_file, box):
        frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
        self.pipeline.init(frame, box)

    def track(self, image_file):
        frame = cv2.imread(image_file, cv2.IMREAD_COLOR)
        rect_pred = self.pipeline.update(frame)
        return rect_pred
