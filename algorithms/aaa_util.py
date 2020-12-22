import sys
import numpy as np

import graph_tool
import graph_tool.topology
import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import scipy.special as sc

feature_factor = 5


def calc_overlap(rect1, rect2):
    r"""Generalized Intersection over Union
    https://giou.stanford.edu/
    """
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


def calc_similarity(feature1, feature2):
    with torch.no_grad():
        if feature1.ndim == 1:
            feature1 = feature1[None, :]
        if feature2.ndim == 1:
            feature2 = feature2[None, :]

        dot_product = torch.matmul(feature1, feature2.T)
        norm_feature1 = torch.norm(feature1, dim=1, keepdim=True)
        norm_feature2 = torch.norm(feature2, dim=1, keepdim=True).T
        sim = dot_product / norm_feature1 / norm_feature2
        score = (1 + sim) / 2
    return score.cpu().numpy()


class WAADelayed:
    def __init__(self):
        self.init_w = True

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

        self.real_D += gradient_losses.shape[1]

        for i in range(1, gradient_losses.shape[1] + 1):
            self.real_D += i
            if self.est_D < self.real_D:
                self.est_D *= 2
                if self.init_w:
                    self.w = np.ones(len(self.w)) / len(self.w)

        lr = np.sqrt(self.est_D * np.log(len(self.w)))

        changes = lr * gradient_losses.sum(axis=1)
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))


class AnchorDetector:
    def __init__(self, threshold=0.0):
        # Threshold for detecting anchor frame
        self.threshold = threshold

    def init(self, target_feature):
        self.target_feature = target_feature

    def detect(self, features):
        feature_scores = (
            calc_similarity(self.target_feature, features)[0] ** feature_factor
        )
        detected = np.where(feature_scores >= self.threshold)[0]
        return detected, feature_scores


class ShortestPathTracker:
    def __init__(self):
        self.g = graph_tool.Graph()
        self.g.edge_properties["cost"] = self.g.new_edge_property("int")

    def initialize(self, box, target_feature):
        self.frame_id = -1

        self.g.clear()
        self.source = self.g.add_vertex()
        self.sink = self.g.add_vertex()

        self.prev_boxes = box[None, :]
        self.prev_features = target_feature[None, :]
        self.prev_feature_scores = np.array([1])

    def get_vertex_id(self, frame, idx):
        return frame * 100 + idx + 2

    def track(self, curr_boxes, curr_features, curr_feature_scores, f2i_factor=10000):
        self.frame_id += 1
        prev_frame_id = self.frame_id - 1

        prob_features = (
            calc_similarity(self.prev_features, curr_features) ** feature_factor
        )
        prob_feature_scores = prob_features * curr_feature_scores
        for prev_idx in range(len(self.prev_boxes)):
            prev_vertex = (
                self.source
                if prev_frame_id == -1
                else self.get_vertex_id(prev_frame_id, prev_idx)
            )

            prob_ious = calc_overlap(self.prev_boxes[prev_idx], curr_boxes)
            costs = -np.log(prob_ious * prob_feature_scores[prev_idx, :] + 1e-7)
            costs = (costs * f2i_factor).astype(int)
            for curr_idx in range(len(curr_boxes)):
                edge = self.g.add_edge(
                    prev_vertex, self.get_vertex_id(self.frame_id, curr_idx)
                )
                self.g.edge_properties["cost"][edge] = costs[curr_idx]

        self.prev_boxes = curr_boxes
        self.prev_features = curr_features
        self.prev_feature_scores = curr_feature_scores

    def run(self):
        for last_idx in range(len(self.prev_boxes)):
            edge = self.g.add_edge(
                self.get_vertex_id(self.frame_id, last_idx), self.sink
            )
            self.g.edge_properties["cost"][edge] = 0

        path, _ = graph_tool.topology.shortest_path(
            self.g,
            self.source,
            self.sink,
            weights=self.g.edge_properties["cost"],
            dag=True,
        )
        ids = []
        for i in path[1:-1]:
            vertex_id = int(i) - 2
            frame_id = vertex_id // 100
            idx = vertex_id % 100
            ids.append([frame_id, idx])
        return ids


class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        model = models.resnet50(pretrained=True)
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
            croped_image = image.crop((min_x, min_y, max_x, max_y))
            croped_image = self.transform(croped_image)
            croped_images.append(croped_image)
        croped_images = torch.stack(croped_images)

        with torch.no_grad():
            croped_images = croped_images.to(self.device)
            features = (
                self.extractor(croped_images).view(croped_images.shape[0], -1).detach()
            )
        return features
