import sys
import numpy as np
from base_tracker import BaseTracker

sys.path.append("external/pyCFTrackers")
sys.path.append("external/pysot-toolkit")
from pysot.utils import overlap_ratio
from cftracker.config.mccth_staple_config import MCCTHOTBConfig


class Expert:
    def __init__(self):
        self.rect_positions = []
        self.centers = []
        self.smooth_scores = []


class MCCT(BaseTracker):
    def __init__(self, n_experts, mode, mu):
        super(MCCT, self).__init__(f"MCCT_{mode}_{mu:.2f}")

        self.period = MCCTHOTBConfig().period
        self.expert_num = n_experts
        self.mu = mu

    def initialize(self, image_file, box):
        weight_num = np.arange(self.period)
        self.weight = 1.1 ** weight_num
        self.psr_score = [0]
        self.id_ensemble = np.ones(self.expert_num)
        self.frame_idx = 0
        self.experts = [Expert() for _ in range(self.expert_num)]

        center = box[:2] + box[2:] / 2
        for i in range(self.expert_num):
            self.experts[i].rect_positions.append(box)
            self.experts[i].smooth_scores.append(1)
            self.experts[i].centers.append(center)

    def track(self, image_file, boxes):
        self.frame_idx += 1

        for i in range(self.expert_num):
            center = boxes[i][:2] + boxes[i][2:] / 2
            pre_center = self.experts[i].centers[-1]
            self.experts[i].rect_positions.append(boxes[i])
            self.experts[i].centers.append(center)

            smooth = np.linalg.norm(center - pre_center)
            avg_dim = np.sum(boxes[i][2:]) / 2
            self.experts[i].smooth_scores.append(np.exp(-((smooth / avg_dim) ** 2) / 2))

        if self.frame_idx >= self.period - 1:
            for i in range(self.expert_num):
                self.id_ensemble[i] = self.robustness_eva(i)

            idx = np.argmax(self.id_ensemble)
            self.box = boxes[idx]
        else:
            self.box = boxes[0]

        return (self.box, [self.box], self.id_ensemble)

    def robustness_eva(self, num):
        overlap_score = np.zeros((self.period, self.expert_num))
        src_bboxes = np.array(self.experts[num].rect_positions[-self.period :])
        for i in range(self.expert_num):
            target_bboxes = np.array(self.experts[i].rect_positions[-self.period :])
            overlaps = overlap_ratio(src_bboxes, target_bboxes)
            overlap_score[:, i] = np.exp(-((1 - overlaps) ** 2))
        avg_overlap = np.mean(overlap_score, axis=1)
        expert_avg_overlap = np.mean(overlap_score, axis=0)
        var_overlap = np.sqrt(
            np.mean((overlap_score - expert_avg_overlap[np.newaxis, :]) ** 2, axis=1)
        )
        norm_factor = 1 / np.sum(self.weight)
        weight_avg_overlap = norm_factor * (self.weight.dot(avg_overlap))
        weight_var_overlap = norm_factor * (self.weight.dot(var_overlap))
        pair_score = weight_avg_overlap / (weight_var_overlap + 0.008)

        smooth_score = self.experts[num].smooth_scores[-self.period :]
        self_score = norm_factor * self.weight.dot(smooth_score)

        reliability = self.mu * pair_score + (1 - self.mu) * self_score
        return reliability
