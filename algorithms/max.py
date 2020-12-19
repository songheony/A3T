import torch
import numpy as np
from PIL import Image
from base_tracker import BaseTracker
from .aaa_util import FeatureExtractor, AnchorDetector


class Max(BaseTracker):
    def __init__(self, n_experts):
        super(Max, self).__init__("Max")

        self.n_experts = n_experts

        # Anchor extractor
        self.detector = AnchorDetector(threshold=0.0)

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor = FeatureExtractor(device)

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")

        # Extract target image
        self.target_feature = self.extractor.extract(image, [box])[0]

        # Init detector with target feature
        self.detector.init(self.target_feature)

    def track(self, image_file, boxes):
        image = Image.open(image_file).convert("RGB")

        # Extract features from boxes
        features = self.extractor.extract(image, boxes)

        # Detect if it is anchor frame
        detected, feature_scores = self.detector.detect(features)

        # Get the index of maximum feature score
        max_id = np.argmax(feature_scores)
        weight = np.zeros((len(boxes)))
        weight[max_id] = 1

        return boxes[max_id], [boxes[max_id]], weight
