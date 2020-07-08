import numpy as np

import sys
import pickle

from algorithms.aaa_util import calc_overlap

sys.path.append("external/pysot-toolkit/pysot")
sys.path.append("external/pytracking")

from pytracking.evaluation.environment import env_settings
from utils import success_overlap, success_error


class OfflineBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array(
            [
                (bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2),
            ]
        ).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh + 1e-16)

    def eval_anchor_frame(self, eval_algorithm, eval_trackers):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        thresholds = np.arange(0, 51, 1)

        anchor_frame = {}
        anchor_success = {}
        anchor_precision = {}

        anchor_success_ = {}
        anchor_precision_ = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            results_dir = "{}/{}".format(env_settings().results_path, eval_algorithm)
            base_results_path = "{}/{}".format(results_dir, seq.name)
            offline_path = "{}_offline.pkl".format(base_results_path)
            with open(offline_path, "rb") as fp:
                offline_bb = pickle.load(fp)
            offline_bb.insert(0, [gt_traj[0]])
            valid_idx = [x is not None for x in offline_bb]
            anchor_frame[seq.name] = valid_idx

            anchor_box = np.array(
                [offline_bb[i][-1] for i in range(len(offline_bb)) if valid_idx[i]]
            )
            anchor_gt = np.array(
                [gt_traj[i] for i in range(len(gt_traj)) if valid_idx[i]]
            )
            if len(anchor_box) > 0:
                anchor_success_[seq.name] = success_overlap(
                    anchor_gt, anchor_box, len(anchor_gt)
                )

                anchor_gt_center = self.convert_bb_to_center(anchor_gt)
                results_center = self.convert_bb_to_center(anchor_box)
                anchor_precision_[seq.name] = success_error(
                    anchor_gt_center, results_center, thresholds, len(anchor_gt_center)
                )
            else:
                anchor_success_[seq.name] = np.nan
                anchor_precision_[seq.name] = np.nan
        anchor_success[eval_algorithm] = anchor_success_
        anchor_precision[eval_algorithm] = anchor_precision_

        for tracker_name in eval_trackers:
            anchor_success_ = {}
            anchor_precision_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")

                valid_idx = anchor_frame[seq.name]
                anchor_gt = np.array(
                    [gt_traj[i] for i in range(len(gt_traj)) if valid_idx[i]]
                )
                anchor_box = np.array(
                    [tracker_traj[i] for i in range(len(tracker_traj)) if valid_idx[i]]
                )
                if len(anchor_box) > 0:
                    anchor_success_[seq.name] = success_overlap(
                        anchor_gt, anchor_box, len(anchor_gt)
                    )

                    anchor_gt_center = self.convert_bb_to_center(anchor_gt)
                    results_center = self.convert_bb_to_center(anchor_box)
                    anchor_precision_[seq.name] = success_error(
                        anchor_gt_center,
                        results_center,
                        thresholds,
                        len(anchor_gt_center),
                    )
                else:
                    anchor_success_[seq.name] = np.nan
                    anchor_precision_[seq.name] = np.nan
            anchor_success[tracker_name] = anchor_success_
            anchor_precision[tracker_name] = anchor_precision_
        return anchor_frame, anchor_success, anchor_precision

    def eval_offline_tracker(self, eval_algorithm, eval_trackers):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        thresholds = np.arange(0, 51, 1)

        frame_length = {}
        success_ret = {}
        precision_ret = {}

        success_ret_ = {}
        precision_ret_ = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            results_dir = "{}/{}".format(env_settings().results_path, eval_algorithm)
            base_results_path = "{}/{}".format(results_dir, seq.name)
            offline_path = "{}_offline.pkl".format(base_results_path)
            with open(offline_path, "rb") as fp:
                offline_bb = pickle.load(fp)

            results = [gt_traj[0]]
            for box in offline_bb:
                if box is not None:
                    results += box.tolist()
            results = np.array(results)
            offline_gt = gt_traj[: len(results)]
            frame_length[seq.name] = len(results)

            if len(results) > 0:
                success_ret_[seq.name] = success_overlap(
                    offline_gt, results, len(results)
                )

                offline_gt_center = self.convert_bb_to_center(offline_gt)
                results_center = self.convert_bb_to_center(results)
                precision_ret_[seq.name] = success_error(
                    offline_gt_center, results_center, thresholds, len(results)
                )
            else:
                success_ret_[seq.name] = np.nan
                precision_ret_[seq.name] = np.nan
        success_ret[eval_algorithm] = success_ret_
        precision_ret[eval_algorithm] = precision_ret_

        for tracker_name in eval_trackers:
            success_ret_ = {}
            precision_ret_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")
                valid_gt = gt_traj[: frame_length[seq.name]]
                valid_results = tracker_traj[: frame_length[seq.name]]

                if frame_length[seq.name] > 0:
                    success_ret_[seq.name] = success_overlap(
                        valid_gt, valid_results, frame_length[seq.name]
                    )

                    valid_gt_center = self.convert_bb_to_center(valid_gt)
                    valid_results_center = self.convert_bb_to_center(valid_results)
                    precision_ret_[seq.name] = success_error(
                        valid_gt_center,
                        valid_results_center,
                        thresholds,
                        frame_length[seq.name],
                    )
                else:
                    success_ret_[seq.name] = np.nan
                    precision_ret_[seq.name] = np.nan
            success_ret[tracker_name] = success_ret_
            precision_ret[tracker_name] = precision_ret_

        return success_ret, precision_ret

    def eval_regret(self, eval_algorithm, eval_trackers, experts):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        regret_gt = {}
        regret_offline = {}
        for tracker_name in eval_trackers:
            regret_gt_ = {}
            regret_offline_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)

                # get offline
                results_dir = "{}/{}".format(
                    env_settings().results_path, eval_algorithm
                )
                base_results_path = "{}/{}".format(results_dir, seq.name)
                offline_path = "{}_offline.pkl".format(base_results_path)
                with open(offline_path, "rb") as fp:
                    offline_bb = pickle.load(fp)
                offline_results = [gt_traj[0]]
                for box in offline_bb:
                    if box is not None:
                        offline_results += box.tolist()
                offline_results = np.array(offline_results)

                # get results
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")

                regret_gt_[seq.name] = np.sum(1 - calc_overlap(gt_traj, tracker_traj))

                valid_results = tracker_traj[: len(offline_results)]
                if len(offline_results) > 0:
                    regret_offline_[seq.name] = np.sum(
                        1 - calc_overlap(offline_results, valid_results)
                    )
                else:
                    regret_offline_[seq.name] = np.nan
            regret_gt[tracker_name] = regret_gt_
            regret_offline[tracker_name] = regret_offline_

        for seq in self.dataset:
            min_gt = np.min(
                [regret_gt[expert_name][seq.name] for expert_name in experts]
            )
            min_offline = np.min(
                [regret_offline[expert_name][seq.name] for expert_name in experts]
            )

            for tracker_name in eval_trackers:
                regret_gt[tracker_name][seq.name] -= min_gt
                regret_offline[tracker_name][seq.name] -= min_offline

        return regret_gt, regret_offline
