import sys
import numpy as np
import torch
import scipy.special as sc
from .algorithm import Algorithm
from .utils import Extractor, MinCostFlowTracker, calc_overlap, cosine_similarity


def calc_cost_link(self, info1, info2, eps=1e-7):
    rect1 = info1["rect"]
    feature1 = info1["feature"]
    rect2 = info2["rect"]
    feature2 = info2["feature"]
    prob_iou = calc_overlap(rect1, rect2)
    prob_feature = cosine_similarity(feature1, feature2)

    prob_sim = prob_iou * prob_feature
    return -np.log(prob_sim + eps)


class WAADelayed:
    def __init__(self):
        pass

    def init(self, n):
        self.w = np.ones(n) / n
        self.est_D = 1
        self.real_D = 0

    """
    gradient_losses should be n * len(dt)
    """

    def update(self, gradient_losses):
        # check the number of element
        assert gradient_losses.shape[0] == len(self.w)

        for i in range(1, gradient_losses.shape[1] + 1):
            self.real_D += i
            if self.est_D < self.real_D:
                self.est_D *= 2

        lr = np.sqrt(self.est_D * np.log(len(self.w)))

        changes = lr * gradient_losses.sum(axis=1)
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))


class AAA(Algorithm):
    def __init__(self, n_experts):
        super(Algorithm, self).__init__("Ours")

        # Threshold for detecting anchor frame
        self.threshold = 0.74

        # The number of experts
        self.n_experts = n_experts

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        self.extractor = Extractor(device)

        # Offline tracker
        self.offline = MinCostFlowTracker(calc_cost_link)

        # Online learner
        self.learner = WAADelayed()

    def init(self, image, box):
        # Previous boxes of experts
        self.prev_boxes = []

        # Extract target image
        self.target_feature = self.extractor.extract(image, [box])[0]

        # Init detections with target feature
        self.detections = [[{"rect": box, "feature": self.target_feature}]]

        # Init online learner
        self.learner.init(self.n_experts)

    def update(self, image, boxes):
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
            self.detections.append(
                [{"rect": boxes[i], "feature": features[i]} for i in detected]
            )

            # Caluclate optimal path
            self.offline.run(self.detections)

            # Add final box properly
            path = self.offline.ids[:-1] + [
                self.offline.ids[-1][0],
                detected[self.offline.ids[-1][1]],
            ]

            # Get offline tracking results
            self.prev_boxes = np.array(self.prev_boxes)
            offline_results = self.prev_boxes[path]

            # Calc losses of experts
            gradient_losses = self._calc_expert_losses(offline_results)

            # Update weight of experts
            self.learner.update(gradient_losses)

            # Return last box of offline results
            predict = offline_results[-1]

        # Otherwise
        else:
            # Add all boxes to offline tracker
            self.detections.append(
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
