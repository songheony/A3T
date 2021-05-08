import sys
import pickle
import numpy as np
from algorithms.aaa_util import calc_overlap
import path_config

sys.path.append("external/pysot-toolkit")
from pysot.utils import success_overlap, success_error


class OPEBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def convert_bb_to_center(self, bboxes):
        return np.array(
            [
                (bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2),
            ]
        ).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh + 1e-16)

    def get_tracker_traj(self, seq_name, tracker_name):
        results_dir = "{}/{}/{}".format(
            path_config.RESULTS_PATH, tracker_name, self.dataset_name
        )
        base_results_path = "{}/{}".format(results_dir, seq_name)
        results_path = "{}.txt".format(base_results_path)
        tracker_traj = np.loadtxt(results_path, delimiter="\t")
        return tracker_traj

    def get_algorithm_data(self, seq_name, algorithm_name):
        results_dir = "{}/{}/{}".format(
            path_config.RESULTS_PATH, algorithm_name, self.dataset_name
        )
        base_results_path = "{}/{}".format(results_dir, seq_name)
        offline_path = "{}_offline.pkl".format(base_results_path)
        with open(offline_path, "rb") as fp:
            offline_bb = pickle.load(fp)

        weights_path = "{}_weight.txt".format(base_results_path)
        tracker_weight = np.loadtxt(weights_path, delimiter="\t")

        return offline_bb, tracker_weight

    def get_anchor_frames(self, algorithm_name):
        anchor_frames = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            offline_bb, tracker_weight = self.get_algorithm_data(
                seq.name, algorithm_name
            )
            offline_bb.insert(0, [gt_traj[0]])
            anchor_frame = [
                i for i in range(len(offline_bb)) if offline_bb[i] is not None
            ]
            anchor_frames[seq.name] = anchor_frame
        return anchor_frames

    def get_gt_trajs(self):
        gt_trajs = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            gt_trajs[seq.name] = gt_traj

        return gt_trajs

    def eval_times(self, tracker_name):
        time_ret = {}
        for seq in self.dataset:
            results_dir = "{}/{}/{}".format(
                path_config.RESULTS_PATH, tracker_name, self.dataset_name
            )
            base_results_path = "{}/{}".format(results_dir, seq.name)
            times_path = "{}_time.txt".format(base_results_path)
            tracker_time = np.loadtxt(times_path, delimiter="\t", dtype=float)
            time_ret[seq.name] = np.mean(tracker_time[1:])
        return time_ret

    def eval_success(self, tracker_name, anchor_frames=None):
        success_ret = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            valid_idx = ~np.isnan(gt_traj)[:, 0]
            tracker_traj = self.get_tracker_traj(seq.name, tracker_name)

            if anchor_frames is not None:
                anchor_frame_idx = np.zeros((len(gt_traj)), dtype=bool)
                anchor_frame_idx[anchor_frames[seq.name]] = 1
                valid_idx = valid_idx * anchor_frame_idx

            n_frame = sum(valid_idx)

            if n_frame > 0:
                success_ret[seq.name] = success_overlap(
                    gt_traj[valid_idx], tracker_traj[valid_idx], n_frame
                )
            else:
                success_ret[seq.name] = np.nan
        return success_ret

    def eval_precision(self, tracker_name, anchor_frames=None):
        precision_ret = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            valid_idx = ~np.isnan(gt_traj)[:, 0]
            tracker_traj = self.get_tracker_traj(seq.name, tracker_name)

            if anchor_frames is not None:
                anchor_frame_idx = np.zeros((len(gt_traj)), dtype=bool)
                anchor_frame_idx[anchor_frames[seq.name]] = 1
                valid_idx = valid_idx * anchor_frame_idx

            n_frame = sum(valid_idx)

            if n_frame > 0:
                gt_center = self.convert_bb_to_center(gt_traj[valid_idx])
                tracker_center = self.convert_bb_to_center(tracker_traj[valid_idx])
                thresholds = np.arange(0, 51, 1)
                precision_ret[seq.name] = success_error(
                    gt_center, tracker_center, thresholds, n_frame
                )
            else:
                precision_ret[seq.name] = np.nan
        return precision_ret

    def eval_norm_precision(self, tracker_name, anchor_frames=None):
        norm_precision_ret = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            valid_idx = ~np.isnan(gt_traj)[:, 0]
            tracker_traj = self.get_tracker_traj(seq.name, tracker_name)

            if anchor_frames is not None:
                anchor_frame_idx = np.zeros((len(gt_traj)), dtype=bool)
                anchor_frame_idx[anchor_frames[seq.name]] = 1
                valid_idx = valid_idx * anchor_frame_idx

            n_frame = sum(valid_idx)

            if n_frame > 0:
                gt_center_norm = self.convert_bb_to_norm_center(
                    gt_traj[valid_idx], gt_traj[valid_idx, 2:4]
                )
                tracker_center_norm = self.convert_bb_to_norm_center(
                    tracker_traj[valid_idx], gt_traj[valid_idx, 2:4]
                )
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret[seq.name] = success_error(
                    gt_center_norm, tracker_center_norm, thresholds, n_frame
                )
            else:
                norm_precision_ret[seq.name] = np.nan
        return norm_precision_ret

    def eval_loss(self, algorithm_name, tracker_name):
        error_ret = {}
        loss_ret = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            valid_idx = ~np.isnan(gt_traj)[:, 0]
            tracker_traj = self.get_tracker_traj(seq.name, tracker_name)

            offline_bb, tracker_weight = self.get_algorithm_data(
                seq.name, algorithm_name
            )

            # flat offilne results
            offline_results = [gt_traj[0]]
            for box in offline_bb:
                if box is not None:
                    if isinstance(box, np.ndarray):
                        offline_results += box.tolist()
                    else:
                        offline_results += box
            offline_results = np.array(offline_results)

            # calc
            error_ret[seq.name] = 1 - calc_overlap(gt_traj[valid_idx], tracker_traj[valid_idx])
            valid_results = tracker_traj[: len(offline_results)]
            if len(offline_results) > 0:
                loss_ret[seq.name] = 1 - calc_overlap(offline_results, valid_results)
            else:
                loss_ret[seq.name] = np.nan

        return error_ret, loss_ret

    def eval_offline(self, algorithm_name, tracker_name):
        success_ret = {}
        precision_ret = {}
        for seq in self.dataset:
            gt_traj = np.array(seq.ground_truth_rect)
            offline_bb, tracker_weight = self.get_algorithm_data(
                seq.name, algorithm_name
            )
            tracker_traj = self.get_tracker_traj(seq.name, tracker_name)

            # flat offilne results
            offline_results = [gt_traj[0]]
            for box in offline_bb:
                if box is not None:
                    if isinstance(box, np.ndarray):
                        offline_results += box.tolist()
                    else:
                        offline_results += box
            offline_results = np.array(offline_results)

            valid_gt_traj = gt_traj[:len(offline_results)]
            valid_tracker_traj = tracker_traj[:len(offline_results)]
            valid_idx = ~np.isnan(valid_gt_traj)[:, 0]
            n_frame = sum(valid_idx)

            if algorithm_name == tracker_name:
                success_ret[seq.name] = success_overlap(
                    valid_gt_traj[valid_idx], offline_results[valid_idx], n_frame
                )
                gt_center = self.convert_bb_to_center(valid_gt_traj[valid_idx])
                tracker_center = self.convert_bb_to_center(offline_results[valid_idx])
                thresholds = np.arange(0, 51, 1)
                precision_ret[seq.name] = success_error(
                    gt_center, tracker_center, thresholds, n_frame
                )
            else:
                success_ret[seq.name] = success_overlap(
                    valid_gt_traj[valid_idx], valid_tracker_traj[valid_idx], n_frame
                )
                gt_center = self.convert_bb_to_center(valid_gt_traj[valid_idx])
                tracker_center = self.convert_bb_to_center(valid_tracker_traj[valid_idx])
                thresholds = np.arange(0, 51, 1)
                precision_ret[seq.name] = success_error(
                    gt_center, tracker_center, thresholds, n_frame
                )
        return success_ret, precision_ret
