import torch
import random
import numpy as np
from PIL import Image
from base_tracker import BaseTracker
from .aaa_util import (
    FeatureExtractor,
    ShortestPathTracker,
    WAADelayed,
    AnchorDetector,
    calc_overlap,
)


class AAA(BaseTracker):
    def __init__(
        self, n_experts, mode="SuperFast", threshold=0.0,
    ):
        super(AAA, self).__init__(
            f"AAA/{mode}/{threshold:.2f}" if threshold > 0 else f"WithoutDelay/{mode}"
        )

        # The number of experts
        self.n_experts = n_experts

        # Anchor extractor
        self.detector = AnchorDetector(threshold=threshold)

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor = FeatureExtractor(device)

        # Offline tracker
        self.offline = ShortestPathTracker()

        # If offline tracker is reset
        self.reset_offline = True

        # Online learner
        self.learner = WAADelayed()

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")

        # Previous boxes of experts
        self.prev_boxes = []

        # Extract target image
        self.target_feature = self.extractor.extract(image, [box])

        # Init detector with target feature
        self.detector.init(self.target_feature)

        # Init offline tracker with target feature
        self.offline.initialize(box, self.target_feature)

        # Init online learner
        self.learner.init(self.n_experts)

    def track(self, image_file, boxes):
        image = Image.open(image_file).convert("RGB")

        # Save box of experts
        self.prev_boxes.append(boxes)

        # Extract features from boxes
        features = self.extractor.extract(image, boxes)

        # Detect if it is anchor frame
        detected, feature_scores = self.detector.detect(features)
        anchor = len(detected) > 0

        # If it is anchor frame,
        if anchor:
            # Add only boxes whose score is over than threshold to offline tracker
            self.offline.track(
                boxes, features, feature_scores
            )

            # Caluclate optimal path
            path = self.offline.run(detected)

            # Get the last box's id
            final_box_id = path[-1][1]

            # Change to ndarray
            self.prev_boxes = np.stack(self.prev_boxes)

            if self.reset_offline:
                # Reset offline tracker
                self.offline.initialize(boxes[final_box_id], features[final_box_id])

                # Get offline tracking results
                offline_results = np.array(
                    [self.prev_boxes[frame, ind[1]] for frame, ind in enumerate(path)]
                )

            else:
                offline_results = np.array(
                    [self.prev_boxes[frame, ind[1]] for frame, ind in enumerate(path[-len(self.prev_boxes):])]
                )

            # Calc losses of experts
            gradient_losses = self._calc_expert_losses(offline_results)

            # Clean previous boxes
            self.prev_boxes = []

            # Update weight of experts
            self.learner.update(gradient_losses)

            # Return last box of offline results
            predict = boxes[final_box_id]

        # Otherwise
        else:
            # Add all boxes to offline tracker
            self.offline.track(boxes, features, feature_scores)

            # No offline result here
            offline_results = None

            # Return box with aggrogating experts' box
            predict = random.choices(boxes, weights=self.learner.w)[0]

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
