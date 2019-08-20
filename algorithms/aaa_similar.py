import torch
import numpy as np
from PIL import Image
from .utils import calc_overlap, cosine_similarity, calc_cost_link
from .algorithm import Algorithm
from .aaa import Extractor, ShortestPathTracker, WAADelayed


class AAA_similar(Algorithm):
    def __init__(
        self,
        n_experts,
        threshold,
        check_dist=False,
        check_feature=True,
        check_target=False,
    ):
        super(AAA_similar, self).__init__(
            "AAA_similar_%0.2f_%s_%s_%s"
            % (threshold, check_dist, check_feature, check_target)
        )

        # Threshold for detecting anchor frame
        self.threshold = threshold

        # Whether check distance for cost
        self.check_dist = check_dist

        # Whether check feature for cost
        self.check_feature = check_feature

        # Whether check target feature for cost
        self.check_target = check_target

        # Whether reset offline tracker
        self.reset_offline = True

        # The number of experts
        self.n_experts = n_experts

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        self.extractor = Extractor(device)

        # Offline tracker
        self.offline = ShortestPathTracker(self.cost_function)

        # Online learner
        self.learner = WAADelayed()

    def cost_function(self, info1, info2):
        return calc_cost_link(
            self.target_feature,
            info1,
            info2,
            check_dist=self.check_dist,
            check_feature=self.check_feature,
            check_target=self.check_target,
        )

    def initialize(self, image, box):
        image = Image.fromarray(image)

        # Previous boxes of experts
        self.prev_boxes = []

        # Extract target image
        self.target_feature = self.extractor.extract(image, [box])[0]

        # Init offline tracker with target feature
        self.offline.initialize({"rect": box, "feature": self.target_feature})

        # Init online learner
        self.learner.init(self.n_experts)

    def track(self, image, boxes):
        image = Image.fromarray(image)

        # Save box of experts
        self.prev_boxes.append(boxes)

        # Extract features from boxes
        features = self.extractor.extract(image, boxes)

        # Detect if it is anchor frame
        detected = self._detect_anchor(features)
        anchor = len(detected) > 0

        # If it is anchor frame,
        if anchor:
            # Add only boxes whose score is over than threshold to offline tracker
            self.offline.track(
                [{"rect": boxes[i], "feature": features[i]} for i in detected],
                is_last=True,
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
                    {"rect": boxes[final_box_id], "feature": features[final_box_id]}
                )
            else:
                # Get only unevaluated frames' boxes
                path = path[-len(self.prev_boxes) :]

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
                    {"rect": box, "feature": feature}
                    for box, feature in zip(boxes, features)
                ]
            )

            # No offline result here
            offline_results = None

            # Return box with aggrogating experts' box
            predict = np.dot(self.learner.w, boxes)

        return predict, offline_results

    def _detect_anchor(self, features):
        detected = []
        for i, feature in enumerate(features):
            score = cosine_similarity(self.target_feature, feature)
            if score >= self.threshold:
                detected.append(i)
        return detected

    def _calc_expert_losses(self, offline_results):
        """
        offline_results = #frames X 4
        """

        expert_gradient_losses = np.zeros((self.n_experts, len(offline_results)))

        for i in range(self.n_experts):
            expert_results = self.prev_boxes[:, i, :]
            expert_gradient_losses[i] = calc_overlap(expert_results, offline_results)

        return expert_gradient_losses
