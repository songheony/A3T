import sys
import cv2
import torch
from base_tracker import BaseTracker
import path_config

sys.path.append("external/pysot")
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


class SiamRPNPPGroup(BaseTracker):
    def __init__(self, backbone, target):
        super(SiamRPNPPGroup, self).__init__(f"SiamRPN++Group/{backbone}/{target}")

        if backbone == "AlexNet" and target == "OTB":
            config = path_config.SIAMRPNPP_ALEXNET_OTB_CONFIG
            snapshot = path_config.SIAMRPNPP_ALEXNET_OTB_SNAPSHOT
        elif backbone == "AlexNet" and target == "VOT":
            config = path_config.SIAMRPNPP_ALEXNET_CONFIG
            snapshot = path_config.SIAMRPNPP_ALEXNET_SNAPSHOT
        elif backbone == "ResNet-50" and target == "OTB":
            config = path_config.SIAMRPNPP_RESNET_OTB_CONFIG
            snapshot = path_config.SIAMRPNPP_RESNET_OTB_SNAPSHOT
        elif backbone == "ResNet-50" and target == "VOT":
            config = path_config.SIAMRPNPP_RESNET_CONFIG
            snapshot = path_config.SIAMRPNPP_RESNET_SNAPSHOT
        elif backbone == "ResNet-50" and target == "VOTLT":
            config = path_config.SIAMRPNPP_RESNET_LT_CONFIG
            snapshot = path_config.SIAMRPNPP_RESNET_LT_SNAPSHOT
        elif backbone == "MobileNetV2" and target == "VOT":
            config = path_config.SIAMRPNPP_MOBILENET_CONFIG
            snapshot = path_config.SIAMRPNPP_MOBILENET_SNAPSHOT
        elif backbone == "SiamMask" and target == "VOT":
            config = path_config.SIAMPRNPP_SIAMMASK_CONFIG
            snapshot = path_config.SIAMPRNPP_SIAMMASK_SNAPSHOT
        else:
            raise ValueError("Invalid backbone and target")

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

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = cv2.imread(image_file)
        bbox = self.tracker.track(image)["bbox"]
        return bbox
