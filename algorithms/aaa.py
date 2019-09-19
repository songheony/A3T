import torch
import numpy as np
from PIL import Image
from .algorithm import Algorithm
from .aaa_util import (
    FeatureExtractor,
    ShortestPathTracker,
    WAADelayed,
    AnchorDetector,
    calc_overlap,
    calc_iou_score,
)


class AAA(Algorithm):
    def __init__(
        self,
        n_experts,
        iou_threshold=0.0,
        feature_threshold=0.0,
        reset_target=True,
        only_max=True,
        use_iou=True,
        use_feature=True,
        cost_iou=True,
        cost_feature=True,
        cost_score=True,
    ):
        super(AAA, self).__init__(
            "AAA_%s_%s_%s_%s_%s_%s_%s_%s_%s"
            % (
                iou_threshold,
                feature_threshold,
                reset_target,
                only_max,
                use_iou,
                use_feature,
                cost_iou,
                cost_feature,
                cost_score,
            )
        )

        # Whether reset target feature
        self.reset_target = reset_target

        # Whether reset offline tracker
        self.reset_offline = True

        # The number of experts
        self.n_experts = n_experts

        # Anchor extractor
        self.detector = AnchorDetector(
            iou_threshold=iou_threshold,
            feature_threshold=feature_threshold,
            only_max=only_max,
            use_iou=use_iou,
            use_feature=use_feature,
            cost_iou=cost_iou,
            cost_feature=cost_feature,
            cost_score=cost_score,
        )

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor = FeatureExtractor(device)

        # Offline tracker
        self.offline = ShortestPathTracker(self.detector.cost_function)

        # Online learner
        self.learner = WAADelayed()

    def initialize(self, image, box):
        image = Image.fromarray(image)

        # Previous boxes of experts
        self.prev_boxes = []

        # Extract target image
        if self.detector.use_feature or self.detector.cost_feature:
            self.target_feature = self.extractor.extract(image, [box])[0]
        else:
            self.target_feature = None

        # Init detector with target feature
        self.detector.init(self.target_feature)

        # Init offline tracker with target feature
        self.offline.initialize(
            {"rect": box, "feature": self.target_feature, "iou_score": 1}
        )

        # Init online learner
        self.learner.init(self.n_experts)

    def track(self, image, boxes):
        image = Image.fromarray(image)

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
            # Add only boxes whose score is over than threshold to offline tracker
            self.offline.track(
                [
                    {
                        "rect": boxes[i],
                        "feature": features[i],
                        "iou_score": iou_scores[i],
                    }
                    for i in detected
                ]
            )

            # Caluclate optimal path
            path = self.offline.run()

            # Get the last box's id
            final_box_id = detected[path[-1][1]]

            # Add final box and cut first box properly
            path = path[1:-1] + [(path[-1][0], final_box_id)]

            if self.reset_offline:
                # Reset offline tracker
                self.offline.initialize(
                    {
                        "rect": boxes[final_box_id],
                        "feature": features[final_box_id],
                        "iou_score": 1,
                    }
                )
            else:
                # Get only unevaluated frames' boxes
                path = path[-len(self.prev_boxes) :]

            if self.reset_target:
                # Init detector with target feature
                self.detector.init(features[final_box_id])

            # Get offline tracking results
            self.prev_boxes = np.array(self.prev_boxes)
            offline_results = np.array(
                [self.prev_boxes[(frame, ind[1])] for frame, ind in enumerate(path)]
            )

            # Calc losses of experts
            gradient_losses = self._calc_expert_losses(offline_results)

            # Update weight of experts
            self.learner.update(gradient_losses)

            # Clean previous boxes
            self.prev_boxes = []

            # Return last box of offline results
            predict = boxes[final_box_id]

        # Otherwise
        else:
            # Add all boxes to offline tracker
            self.offline.track(
                [
                    {"rect": box, "feature": feature, "iou_score": score}
                    for box, feature, score in zip(boxes, features, iou_scores)
                ]
            )

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
