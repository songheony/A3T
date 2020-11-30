import numpy as np
import sys
import path_config

sys.path.append("external/pysot-toolkit/pysot")
from utils import success_overlap, success_error


class OPEBenchmark:
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

    def eval_times(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        time_ret = {}
        for tracker_name in eval_trackers:
            time_ret_ = {}
            for seq in self.dataset:
                results_dir = "{}/{}".format(path_config.RESULTS_PATH, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                times_path = "{}_time.txt".format(base_results_path)
                tracker_time = np.loadtxt(times_path, delimiter="\t", dtype=float)
                time_ret_[seq.name] = np.mean(tracker_time[1:])
            time_ret[tracker_name] = time_ret_
        return time_ret

    def eval_success(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(path_config.RESULTS_PATH, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")
                n_frame = len(gt_traj)
                success_ret_[seq.name] = success_overlap(gt_traj, tracker_traj, n_frame)
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(path_config.RESULTS_PATH, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")
                n_frame = len(gt_traj)
                gt_center = self.convert_bb_to_center(gt_traj)
                tracker_center = self.convert_bb_to_center(tracker_traj)
                thresholds = np.arange(0, 51, 1)
                precision_ret_[seq.name] = success_error(
                    gt_center, tracker_center, thresholds, n_frame
                )
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(path_config.RESULTS_PATH, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")
                n_frame = len(gt_traj)
                gt_center_norm = self.convert_bb_to_norm_center(
                    gt_traj, gt_traj[:, 2:4]
                )
                tracker_center_norm = self.convert_bb_to_norm_center(
                    tracker_traj, gt_traj[:, 2:4]
                )
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[seq.name] = success_error(
                    gt_center_norm, tracker_center_norm, thresholds, n_frame
                )
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret
