from PIL import Image
import numpy as np
import torch
from base_tracker import BaseTracker
from .aaa_util import (
    FeatureExtractor,
    calc_similarity
)


def avgnh(r, c):
    n = r.size
    T = r.copy()
    T[T < 0] = 0
    w = np.exp(T / c)
    total = (1 / n) * np.sum(w) - 2.72
    return total


def find_nh_scale(regrets):
    clower = 1.0
    counter = 0
    while avgnh(regrets, clower) < 0 and counter < 30:
        clower *= 0.5
        counter += 1

    cupper = 1.0
    counter = 0
    while avgnh(regrets, cupper) > 0 and counter < 30:
        cupper *= 2
        counter += 1

    cmid = (cupper + clower) / 2
    counter = 0
    while np.abs(avgnh(regrets, cmid)) > 1e-2 and counter < 30:
        if avgnh(regrets, cmid) > 1e-2:
            clower = cmid
            cmid = (cmid + cupper) / 2
        else:
            cupper = cmid
            cmid = (cmid + clower) / 2
        counter += 1

    return cmid


def nnhedge_weights(r, scale):
    n = r.size
    w = np.zeros((n))

    for i in range(n):
        if r[i] <= 0:
            w[i] = 2.2204e-16
        else:
            w[i] = np.exp(r[i] / scale) / scale

    return w


class HDT(BaseTracker):
    def __init__(self, n_experts, mode, beta):
        super(HDT, self).__init__(f"HDT_{mode}_{beta:.2f}")
        self.n_experts = n_experts
        self.beta = beta

        # Feature extractor
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor = FeatureExtractor(device)

        self.delta_t = 5
        self.scale_gamma = 0.25

    def initialize(self, image_file, box):
        self.frame_idx = -1
        image = Image.open(image_file).convert("RGB")
        self.target_feature = self.extractor.extract(image, [box])[0]

        self.experts_center = np.zeros((self.n_experts, 2))
        self.experts_loss = np.zeros((self.n_experts, self.delta_t))
        self.experts_regret = np.zeros((self.n_experts))
        self.weights = np.ones((1, self.n_experts)) / self.n_experts
        self.center = box[:2] + box[2:] / 2
        self.size = box[2:]

    def track(self, image_file, boxes):
        self.frame_idx += 1

        self.experts_center = boxes[:, :2] + boxes[:, 2:] / 2

        if self.frame_idx > 0:
            self.center = self.weights.dot(self.experts_center)

        image = Image.open(image_file).convert("RGB")
        features = self.extractor.extract(image, boxes)

        distance = np.linalg.norm(self.center - self.experts_center, axis=1)
        distance_loss = distance / np.sum(distance)

        similarity_loss = np.zeros((self.n_experts))
        for i in range(self.n_experts):
            similarity_loss[i] = 1 - calc_similarity(features[i], self.target_feature)

        loss_idx = self.frame_idx % self.delta_t
        self.experts_loss[:, loss_idx] = (1 - self.beta) * similarity_loss + self.beta * distance_loss
        expected_loss = self.weights.dot(self.experts_loss[:, loss_idx])

        mu = np.mean(self.experts_loss, axis=1)
        sigma = np.std(self.experts_loss, axis=1)
        mu[mu < 1e-4] = 0
        sigma[sigma < 1e-4] = 0

        s = (self.experts_loss[:, loss_idx] - mu) / (sigma + 2.2204e-16)
        self.experts_regret += expected_loss - np.tanh(self.scale_gamma * s) * self.experts_loss[:, loss_idx]

        c = find_nh_scale(self.experts_regret)
        self.weights = nnhedge_weights(self.experts_regret, c)
        self.weights /= np.sum(self.weights)

        box = np.zeros((4))
        box[:2] = self.center - self.size / 2
        box[2:] = self.size

        return (box, [box], self.weights)
