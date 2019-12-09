import random
import numpy as np
from base_tracker import BaseTracker


class Random(BaseTracker):
    def __init__(self, n_experts, mode):
        super(Random, self).__init__(f"Random_{mode}")

    def initialize(self, image, box):
        pass

    def track(self, image, boxes):
        id = random.randint(0, len(boxes) - 1)
        weight = np.zeros((len(boxes)))
        weight[id] = 1
        return (boxes[id], [boxes[id]], weight)
