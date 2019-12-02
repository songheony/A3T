import numpy as np
from .algorithm import Algorithm


class Average(Algorithm):
    def __init__(self, n_experts, mode):
        super(Average, self).__init__(f"Average_{mode}")

    def initialize(self, image, box):
        pass

    def track(self, image, boxes):
        return (
            np.mean(boxes, axis=0),
            [np.mean(boxes, axis=0)],
            np.ones((len(boxes))) / len(boxes),
        )
