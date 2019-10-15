import sys
import numpy as np
import cv2
from .algorithm import Algorithm

sys.path.append("external/pyCFTrackers")
from cftracker.scale_estimator import LPScaleEstimator
from cftracker.mccth_staple import cal_ious
from cftracker.config.mccth_staple_config import MCCTHOTBConfig


class Expert:
    def __init__(self):
        self.pos = None
        self.rect_positions = []
        self.centers = []
        self.smoothes = []
        self.smooth_score = None
        self.smooth_scores = []
        self.rob_scores = []


class MCCT(Algorithm):
    def __init__(self, n_experts):
        super(MCCT, self).__init__("MCCT")

        self.n_experts = n_experts

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        config = MCCTHOTBConfig()
        self.scale_adaptation = config.scale_adaptation

        self.period = config.period
        self.expert_num = self.n_experts

        self.scale_config = config.scale_config

        weight_num = np.arange(self.period)
        self.weight = 1.1 ** weight_num
        self.mean_score = [0]
        self.id_ensemble = []
        self.frame_idx = -1
        self.experts = []
        for i in range(self.expert_num):
            self.experts.append(Expert())
            self.id_ensemble.append(1)

        self.frame_idx += 1
        first_frame = image.astype(np.float32)
        bbox = np.array(box).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (x + w / 2, y + h / 2)
        self.w, self.h = w, h
        self.target_sz = (self.w, self.h)

        avg_dim = (w + h) / 2

        if self.scale_adaptation is True:
            self.scale_factor = 1
            self.base_target_sz = self.target_sz
            self.scale_estimator = LPScaleEstimator(
                self.target_sz, config=self.scale_config
            )
            self.scale_estimator.init(
                first_frame, self._center, self.base_target_sz, self.scale_factor
            )

        self.avg_dim = avg_dim
        for i in range(self.expert_num):
            self.experts[i].rect_positions.append(
                [
                    self._center[0] - self.target_sz[0] / 2,
                    self._center[1] - self.target_sz[1] / 2,
                    self.target_sz[0],
                    self.target_sz[1],
                ]
            )
            self.experts[i].rob_scores.append(1)
            self.experts[i].smoothes.append(0)
            self.experts[i].smooth_scores.append(1)
            self.experts[i].centers.append((self._center[0], self._center[1]))

    def track(self, image_file, boxes):
        image = cv2.imread(image_file)
        self.frame_idx += 1
        current_frame = image

        for i in range(self.expert_num):
            x, y, w, h = tuple(boxes[i])
            center = (x + w / 2, y + h / 2)
            self.experts[i].pos = center
            self.experts[i].rect_positions.append(boxes[i])
            self.experts[i].centers.append(center)

            pre_center = self.experts[i].centers[self.frame_idx - 1]
            smooth = np.sqrt(
                (center[0] - pre_center[0]) ** 2 + (center[1] - pre_center[1]) ** 2
            )
            self.experts[i].smoothes.append(smooth)
            self.experts[i].smooth_scores.append(
                np.exp(-smooth ** 2 / (2 * self.avg_dim ** 2))
            )

        if self.frame_idx >= self.period - 1:
            for i in range(self.expert_num):
                self.experts[i].rob_scores.append(
                    self.robustness_eva(
                        self.experts,
                        i,
                        self.frame_idx,
                        self.period,
                        self.weight,
                        self.expert_num,
                    )
                )

                self.id_ensemble[i] = self.experts[i].rob_scores[self.frame_idx]
            self.mean_score.append(np.sum(np.array(self.id_ensemble)) / self.expert_num)
            idx = np.argmax(np.array(self.id_ensemble))
            self._center = self.experts[idx].pos
        else:
            for i in range(self.expert_num):
                self.experts[i].rob_scores.append(1)
            self._center = self.experts[-1].pos
            self.mean_score.append(0)

        if self.scale_adaptation:
            self.scale_factor = self.scale_estimator.update(
                current_frame, self._center, self.base_target_sz, self.scale_factor
            )
            self.target_sz = (
                round(self.base_target_sz[0] * self.scale_factor),
                round(self.base_target_sz[1] * self.scale_factor),
            )

        return (
            [
                self._center[0] - self.target_sz[0] / 2,
                self._center[1] - self.target_sz[1] / 2,
                self.target_sz[0],
                self.target_sz[1],
            ],
            [
                [
                    self._center[0] - self.target_sz[0] / 2,
                    self._center[1] - self.target_sz[1] / 2,
                    self.target_sz[0],
                    self.target_sz[1],
                ]
            ],
            self.id_ensemble,
        )

    def robustness_eva(self, experts, num, frame_idx, period, weight, expert_num):
        overlap_score = np.zeros((period, expert_num))
        for i in range(expert_num):
            bboxes1 = np.array(experts[i].rect_positions)[
                frame_idx - period + 1 : frame_idx + 1
            ]
            bboxes2 = np.array(experts[num].rect_positions)[
                frame_idx - period + 1 : frame_idx + 1
            ]
            overlaps = cal_ious(bboxes1, bboxes2)
            overlap_score[:, i] = np.exp(-(1 - overlaps) ** 2 / 2)
        avg_overlap = np.sum(overlap_score, axis=1) / expert_num
        expert_avg_overlap = np.sum(overlap_score, axis=0) / period
        var_overlap = np.sqrt(
            np.sum((overlap_score - expert_avg_overlap[np.newaxis, :]) ** 2, axis=1)
            / expert_num
        )
        norm_factor = 1 / np.sum(np.array(weight))
        weight_avg_overlap = norm_factor * (weight.dot(avg_overlap))
        weight_var_overlap = norm_factor * (weight.dot(var_overlap))
        pair_score = weight_avg_overlap / (weight_var_overlap + 0.008)
        smooth_score = experts[num].smooth_scores[
            frame_idx - period + 1 : frame_idx + 1
        ]
        self_score = norm_factor * np.sum(np.array(smooth_score) * weight)
        eta = 0.1
        reliability = eta * pair_score + (1 - eta) * self_score
        return reliability
