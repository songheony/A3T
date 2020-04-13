import sys
import numpy as np
from base_tracker import BaseTracker

sys.path.append("external/pyCFTrackers")
from cftracker.mccth_staple import cal_ious
from cftracker.config.mccth_staple_config import MCCTHOTBConfig


class Expert:
    def __init__(self):
        self.rect_positions = []
        self.centers = []
        self.smooth_scores = []
        self.rob_scores = []


class MCCT(BaseTracker):
    def __init__(self, n_experts, mode):
        super(MCCT, self).__init__(f"MCCT_{mode}")

        self.n_experts = n_experts
        self.config = MCCTHOTBConfig()

    def initialize(self, image_file, box):
        self.period = self.config.period
        self.expert_num = self.n_experts

        weight_num = np.arange(self.period)
        self.weight = 1.1 ** weight_num
        self.psr_score = [0]
        self.id_ensemble = []
        self.frame_idx = 0
        self.experts = []
        for i in range(self.expert_num):
            self.experts.append(Expert())
            self.id_ensemble.append(1)

        bbox = np.array(box).astype(np.int64)
        x, y, w, h = tuple(bbox)
        center = (x + w / 2, y + h / 2)

        for i in range(self.expert_num):
            self.experts[i].rect_positions.append(box)
            self.experts[i].rob_scores.append(1)
            self.experts[i].smooth_scores.append(1)
            self.experts[i].centers.append(center)

        self.box = box

    def track(self, image_file, boxes):
        self.frame_idx += 1

        for i in range(self.expert_num):
            x, y, w, h = tuple(boxes[i])
            center = (x + w / 2, y + h / 2)
            self.experts[i].rect_positions.append(boxes[i])
            self.experts[i].centers.append(center)

            pre_center = self.experts[i].centers[-2]
            smooth = np.sqrt(
                (center[0] - pre_center[0]) ** 2 + (center[1] - pre_center[1]) ** 2
            )
            avg_dim = (w + h) / 2
            self.experts[i].smooth_scores.append(
                np.exp(-smooth ** 2 / (2 * avg_dim ** 2))
            )

        if self.frame_idx >= self.period - 1:
            for i in range(self.expert_num):
                rob_score = self.robustness_eva(
                    self.experts, i, self.period, self.weight, self.expert_num
                )
                self.experts[i].rob_scores.append(rob_score)
                self.id_ensemble[i] = rob_score

            idx = np.argmax(np.array(self.id_ensemble))
            self.box = self.experts[idx].rect_positions[-1]
        else:
            for i in range(self.expert_num):
                self.experts[i].rob_scores.append(1)
            self.box = self.experts[0].rect_positions[-1]

        return (self.box, [self.box], self.id_ensemble)

    def robustness_eva(self, experts, num, period, weight, expert_num):
        overlap_score = np.zeros((period, expert_num))
        for i in range(expert_num):
            bboxes1 = np.array(experts[i].rect_positions)[-period:]
            bboxes2 = np.array(experts[num].rect_positions)[-period:]
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
        smooth_score = experts[num].smooth_scores[-period:]
        self_score = norm_factor * np.sum(np.array(smooth_score) * weight)
        eta = 0.1
        reliability = eta * pair_score + (1 - eta) * self_score
        return reliability
