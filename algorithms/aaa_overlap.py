import numpy as np
from PIL import Image
from .utils import calc_overlap, calc_cost_link
from .algorithm import Algorithm
from .aaa import ShortestPathTracker, WAADelayed


class AAA_overlap(Algorithm):
    def __init__(self, n_experts, threshold, check_dist=False):
        super(AAA_overlap, self).__init__(
            "AAA_overlap_%0.2f_%s" % (threshold, check_dist)
        )

        # Threshold for detecting anchor frame
        self.threshold = threshold

        # Whether check distance for cost
        self.check_dist = check_dist

        # Whether reset offline tracker
        self.reset_offline = True

        # The number of experts
        self.n_experts = n_experts

        # Offline tracker
        self.offline = ShortestPathTracker(self.cost_function)

        # Online learner
        self.learner = WAADelayed()

    def cost_function(self, info1, info2):
        return calc_cost_link(
            None,
            info1,
            info2,
            check_dist=self.check_dist,
            check_feature=False,
            check_target=False,
        )

    def initialize(self, image, box):
        image = Image.fromarray(image)

        # Previous boxes of experts
        self.prev_boxes = []

        # Init offline tracker
        self.offline.initialize({"rect": box, "feature": None})

        # Init online learner
        self.learner.init(self.n_experts)

    def track(self, image, boxes):
        image = Image.fromarray(image)

        # Save box of experts
        self.prev_boxes.append(boxes)

        # Detect if it is anchor frame
        detected = self._detect_anchor(boxes)
        anchor = len(detected) > 0

        # If it is anchor frame,
        if anchor:
            # Add only boxes whose score is over than threshold to offline tracker
            self.offline.track(
                [{"rect": boxes[i], "feature": None} for i in detected], is_last=True
            )

            # Caluclate optimal path
            path = self.offline.run()

            # Get the last box's id
            final_box_id = detected[path[-1][1]]

            # Add final box and cut first box properly
            path = path[1:-1] + [(path[-1][0], final_box_id)]

            if self.reset_offline:
                # Reset offline tracker
                self.offline.initialize({"rect": boxes[final_box_id], "feature": None})
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
            self.offline.track([{"rect": box, "feature": None} for box in boxes])

            # No offline result here
            offline_results = None

            # Return box with aggrogating experts' box
            predict = np.dot(self.learner.w, boxes)

        return predict, offline_results

    def _detect_anchor(self, boxes):
        detected = []
        for i, box1 in enumerate(boxes):
            score = []
            for j, box2 in enumerate(boxes):
                if i == j:
                    continue
                score.append(calc_overlap(box1, box2))
            score = np.mean(score)
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
