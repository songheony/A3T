import numpy as np
from .algorithm import Algorithm


class Average(Algorithm):
    def __init__(self):
        super(Average, self).__init__("Average")

    def init(self, image, box):
        pass

    def update(self, image, boxes):
        return np.mean(boxes, axis=0), None
