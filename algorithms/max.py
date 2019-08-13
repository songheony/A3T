import torch
from PIL import Image
from .algorithm import Algorithm
from .utils import Extractor, cosine_similarity


class Max(Algorithm):
    def __init__(self, n_experts):
        super(Max, self).__init__("Max")

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        self.extractor = Extractor(device)

    def initialize(self, image, box):
        image = Image.fromarray(image)
        self.target_feature = self.extractor.extract(image, [box])[0]

    def track(self, image, boxes):
        image = Image.fromarray(image)

        # Extract features from boxes
        features = self.extractor.extract(image, boxes)

        # Detect if it is anchor frame
        max_id = self._detect_anchor(features)

        return boxes[max_id], None

    def _detect_anchor(self, features):
        max_id = -1
        max_score = 0
        for i, feature in enumerate(features):
            score = cosine_similarity(self.target_feature, feature)
            if score > max_score:
                max_id = i
        return max_id
