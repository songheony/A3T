import torch
import numpy as np
from PIL import Image
from .algorithm import Algorithm
from .aaa_util import (
    FeatureExtractor,
    WAADelayed,
    AnchorDetector,
    calc_overlap,
    calc_iou_score,
)


class AAA_gt(Algorithm):
    def __init__(
        self,
        n_experts,
        iou_threshold=0.0,
        feature_threshold=0.0,
        use_iou=True,
        use_feature=True,
    ):
        super(AAA_gt, self).__init__(
            "AAA_gt_%s_%s_%s_%s"
            % (iou_threshold, feature_threshold, use_iou, use_feature)
        )

        # The number of experts
        self.n_experts = n_experts

        # Anchor extractor
        self.detector = AnchorDetector(
            iou_threshold=iou_threshold,
            feature_threshold=feature_threshold,
            only_max=True,
            use_iou=use_iou,
            use_feature=use_feature,
            cost_iou=False,
            cost_feature=False,
            cost_score=False,
        )

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor = FeatureExtractor(device)

        # Online learner
        self.learner = WAADelayed()

    def initialize(self, image, box, gt):
        image = Image.fromarray(image)
        self.anchor = 0
        self.frame = 0
        self.gt = gt

        # Previous boxes of experts
        self.prev_boxes = []

        # Extract target image
        if self.detector.use_feature or self.detector.cost_feature:
            self.target_feature = self.extractor.extract(image, [box])[0]
        else:
            self.target_feature = None

        # Init detector with target feature
        self.detector.init(self.target_feature)

        # Init online learner
        self.learner.init(self.n_experts)

    def track(self, image, boxes):
        image = Image.fromarray(image)
        self.frame += 1

        # Save box of experts
        self.prev_boxes.append(boxes)

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

        # If it is anchor frame,
        if anchor:
            # Get offline tracking results
            self.prev_boxes = np.array(self.prev_boxes)
            offline_results = self.gt[self.anchor + 1 : self.frame].tolist() + [
                boxes[detected[0]]
            ]
            offline_results = np.array(offline_results)
            self.anchor = self.frame

            # Calc losses of experts
            gradient_losses = self._calc_expert_losses(offline_results)

            # Update weight of experts
            self.learner.update(gradient_losses)

            # Clean previous boxes
            self.prev_boxes = []

            # Return last box of offline results
            predict = boxes[detected[0]]

        # Otherwise
        else:
            # No offline result here
            offline_results = None

            # Return box with aggrogating experts' box
            predict = np.dot(self.learner.w, boxes)

        return predict, offline_results, self.learner.w

    def _calc_expert_losses(self, offline_results):
        """
        offline_results = #frames X 4
        """

        expert_gradient_losses = np.zeros((self.n_experts, len(offline_results)))

        for i in range(self.n_experts):
            expert_results = self.prev_boxes[:, i, :]
            expert_gradient_losses[i] = 1 - calc_overlap(
                expert_results, offline_results
            )

        return expert_gradient_losses
