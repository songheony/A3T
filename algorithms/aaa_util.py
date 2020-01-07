import sys
import igraph
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import scipy.special as sc

iou_factor = 1
feature_factor = 5


def calc_overlap(rect1, rect2):
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    left_min = np.minimum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    right_max = np.maximum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    top_min = np.minimum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])
    bottom_max = np.maximum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    closure = np.maximum(0, right_max - left_min) * np.maximum(0, bottom_max - top_min)
    g_iou = iou - (closure - union) / closure
    g_iou = (1 + g_iou) / 2
    g_iou = np.nan_to_num(g_iou)
    return g_iou


def calc_iou_score(boxes):
    scores = []
    for i, box1 in enumerate(boxes):
        score = []
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            score.append(calc_overlap(box1, box2))
        score = np.mean(score)
        scores.append(score)
    return scores


def calc_similarity(ft1, ft2):
    dot_product = np.dot(ft1, ft2)
    norm_ft1 = np.linalg.norm(ft1)
    norm_ft2 = np.linalg.norm(ft2)
    sim = dot_product / (norm_ft1 * norm_ft2)
    return (1 + sim) / 2


class AnchorDetector:
    def __init__(
        self,
        iou_threshold=0.0,
        feature_threshold=0.0,
        only_max=True,
        use_iou=True,
        use_feature=True,
        cost_iou=True,
        cost_feature=True,
        cost_score=True,
    ):
        # IOU threshold for detecting anchor frame
        self.iou_threshold = iou_threshold

        # Feature threshold for detecting anchor frame
        self.feature_threshold = feature_threshold

        # Whether detect only who has the highest score
        self.only_max = only_max

        # Whether check iou for detection
        self.use_iou = use_iou

        # Whether check feature for detection
        self.use_feature = use_feature

        # Whether calculate iou for cost
        self.cost_iou = cost_iou

        # Whether calculate feature for cost
        self.cost_feature = cost_feature

        # Whether calculate score for cost
        self.cost_score = cost_score

    def init(self, target_feature):
        self.target_feature = target_feature

    def detect(self, iou_scores, features):
        if self.only_max:
            max_id = -1
            max_score = 0
            for i, (iou_score, feature) in enumerate(zip(iou_scores, features)):
                flag = False

                if self.use_iou:
                    iou_score = iou_score ** iou_factor
                    flag = flag or (iou_score >= self.iou_threshold)
                else:
                    iou_score = 1.0

                if self.use_feature:
                    feature_score = calc_similarity(self.target_feature, feature)
                    feature_score = feature_score ** feature_factor
                    flag = flag or (feature_score >= self.feature_threshold)
                else:
                    feature_score = 1.0

                score = iou_score * feature_score

                if score > max_score and flag:
                    max_id = i
                    max_score = score
            if max_id != -1:
                detected = [max_id]
            else:
                detected = []
        else:
            detected = []
            for i, (iou_score, feature) in enumerate(zip(iou_scores, features)):
                flag = False

                if self.use_iou:
                    iou_score = iou_score ** iou_factor
                    flag = flag or (iou_score >= self.iou_threshold)

                if self.use_feature:
                    feature_score = calc_similarity(self.target_feature, feature)
                    feature_score = feature_score ** feature_factor
                    flag = flag or (feature_score >= self.feature_threshold)

                if flag:
                    detected.append(i)
        return detected

    def cost_function(self, info1, info2):
        rect1 = info1["rect"]
        rect2 = info2["rect"]
        feature1 = info1["feature"]
        feature2 = info2["feature"]
        iou_score = info2["iou_score"]

        if self.cost_iou:
            prob_iou = calc_overlap(rect1, rect2)[0]
            prob_iou = prob_iou ** iou_factor
        else:
            prob_iou = 1.0

        if self.cost_feature:
            prob_feature = calc_similarity(feature1, feature2)
            prob_feature = prob_feature ** feature_factor
        else:
            prob_feature = 1.0

        if self.cost_score:
            if self.use_iou:
                iou_score = iou_score ** iou_factor
            else:
                iou_score = 1.0
            if self.use_feature:
                feature_score = calc_similarity(self.target_feature, feature2)
                feature_score = feature_score ** feature_factor
            else:
                feature_score = 1.0

            prob_score = iou_score * feature_score
        else:
            prob_score = 1.0

        cost = -np.log(prob_iou * prob_feature * prob_score + 1e-7)
        return cost


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


class ShortestPathTracker:
    def __init__(self, cost_link):
        self._cost_link = cost_link

    def initialize(self, detection):
        self.frame_id = -1
        self.edges = []
        if len(self.edges) == 0:
            self.edges.append(("source", str((self.frame_id, 0)), 0))
        self.prev_detections = [detection]

    def track(self, detections, f2i_factor=10000):
        self.frame_id += 1
        prev_frame_id = self.frame_id - 1
        for i, i_info in enumerate(self.prev_detections):
            for j, j_info in enumerate(detections):
                self.edges.append(
                    (
                        str((prev_frame_id, i)),
                        str((self.frame_id, j)),
                        int(self._cost_link(i_info, j_info) * f2i_factor),
                    )
                )
        self.prev_detections = detections

    def run(self):
        last_edges = self.edges[:]
        for i in range(len(self.prev_detections)):
            last_edges.append((str((self.frame_id, i)), "sink", 0))
        g = igraph.Graph.TupleList(last_edges, weights=True, directed=True)
        path = g.get_shortest_paths(
            "source", to="sink", weights="weight", mode=igraph.OUT, output="vpath"
        )
        ids = []
        for i in path[0][1:-1]:
            ids.append(tuple(int(s) for s in g.vs[i]["name"].strip("()").split(",")))
        return ids


class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        model = models.resnet50(pretrained=True).to(self.device)
        feature_map = list(model.children())
        feature_map.pop()
        self.extractor = nn.Sequential(*feature_map).to(self.device)
        self.extractor.eval()
        self.transform = transforms.Compose(
            [
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def extract(self, image, bboxes):
        features = []
        croped_images = []
        for bbox in bboxes:
            max_x = min(image.size[0], bbox[0] + bbox[2])
            max_y = min(image.size[1], bbox[1] + bbox[3])
            min_x = max(0, bbox[0])
            min_y = max(0, bbox[1])
            croped_image = self.transform(image.crop((min_x, min_y, max_x, max_y)))
            croped_images.append(croped_image)
        croped_images = torch.stack(croped_images)
        features = self._get_vector(croped_images)
        features = features.data.cpu().numpy()
        return features

    def _get_vector(self, x):
        x = x.to(self.device)
        embedding = self.extractor(x).view(x.shape[0], -1).data
        return embedding
