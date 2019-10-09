import torch
import numpy as np
from PIL import Image
from .algorithm import Algorithm
from .aaa_util import FeatureExtractor, AnchorDetector, calc_iou_score


class Baseline(Algorithm):
    def __init__(self, n_experts, name="Baseline", use_iou=True, use_feature=True):
        super(Baseline, self).__init__(name)

        self.n_experts = n_experts

        # Anchor extractor
        self.detector = AnchorDetector(
            iou_threshold=0.0,
            feature_threshold=0.0,
            only_max=True,
            use_iou=use_iou,
            use_feature=use_feature,
            cost_iou=False,
            cost_feature=False,
            cost_score=False,
        )

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        self.extractor = FeatureExtractor(device)

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")

        # Extract target image
        if self.detector.use_feature:
            self.target_feature = self.extractor.extract(image, [box])[0]
        else:
            self.target_feature = None

        self.detector.init(self.target_feature)

    def track(self, image_file, boxes):
        image = Image.open(image_file).convert("RGB")

        # Extract scores from boxes
        if self.detector.use_iou:
            iou_scores = calc_iou_score(boxes)
        else:
            iou_scores = [0] * self.n_experts

        # Extract features from boxes
        if self.detector.use_feature or self.detector.cost_feature:
            features = self.extractor.extract(image, boxes)
        else:
            features = [None] * self.n_experts

        # Detect if it is anchor frame
        detected = self.detector.detect(iou_scores, features)
        anchor = len(detected) > 0

        if anchor:
            max_id = detected[0]
            weight = np.zeros((len(boxes)))
            weight[max_id] = 1

            return boxes[max_id], [boxes[max_id]], weight
        else:
            return np.zeros_like(boxes[0]), None, np.zeros((len(boxes)))
