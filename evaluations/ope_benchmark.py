import numpy as np

from colorama import Style, Fore
import sys
import pickle

sys.path.append("external/pysot-toolkit/pysot")
sys.path.append("external/pytracking")

from pytracking.evaluation.environment import env_settings
from utils import success_overlap, success_error, overlap_ratio


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

    def eval_success(self, eval_trackers, show_anchor=False):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        success_ret = {}
        anchor_frame = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            anchor_frame_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
                base_results_path = "{}/{}".format(results_dir, seq.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t")
                n_frame = len(gt_traj)
                success_ret_[seq.name] = success_overlap(gt_traj, tracker_traj, n_frame)
                if show_anchor:
                    offline_path = "{}_offline.pkl".format(base_results_path)
                    with open(offline_path, "rb") as fp:
                        offline_bb = pickle.load(fp)
                    offline_bb.insert(0, [gt_traj[0]])
                    valid_idx = [x is not None for x in offline_bb]
                    anchor_frame_[seq.name] = valid_idx
                else:
                    anchor_frame_[seq.name] = None
            success_ret[tracker_name] = success_ret_
            anchor_frame[tracker_name] = anchor_frame_
        if show_anchor:
            return success_ret, anchor_frame
        else:
            return success_ret

    def eval_success_offline(self, eval_algorithms):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        anchor_ret = {}
        offline_ret = {}
        anchor_ratio = {}
        for algorithm_name in eval_algorithms:
            anchor_ret_ = {}
            offline_ret_ = {}
            anchor_ratio_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(
                    env_settings().results_path, algorithm_name
                )
                base_results_path = "{}/{}".format(results_dir, seq.name)
                offline_path = "{}_offline.pkl".format(base_results_path)
                with open(offline_path, "rb") as fp:
                    offline_bb = pickle.load(fp)
                offline_bb.insert(0, None)

                results = []
                for box in offline_bb:
                    if box is not None:
                        results += box.tolist()
                results = np.array(results)
                offline_gt = gt_traj[: len(results)]
                if len(results) > 0:
                    offline_ret_[seq.name] = success_overlap(
                        offline_gt, results, len(results)
                    )
                else:
                    offline_ret_[seq.name] = np.array([0])

                valid_idx = [x is not None for x in offline_bb]
                anchor_box = np.array(
                    [offline_bb[i][-1] for i in range(len(offline_bb)) if valid_idx[i]]
                )
                anchor_gt = np.array(
                    [gt_traj[i] for i in range(len(gt_traj)) if valid_idx[i]]
                )
                if len(anchor_box) > 0:
                    anchor_ret_[seq.name] = overlap_ratio(anchor_gt, anchor_box)
                else:
                    anchor_ret_[seq.name] = np.array([0])

                anchor_ratio_[seq.name] = sum(valid_idx) / len(gt_traj)
            offline_ret[algorithm_name] = offline_ret_
            anchor_ret[algorithm_name] = anchor_ret_
            anchor_ratio[algorithm_name] = anchor_ratio_
        return offline_ret, anchor_ret, anchor_ratio

    def eval_precision(self, eval_trackers):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
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

    def eval_precision_offline(self, eval_algorithms):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        thresholds = np.arange(0, 51, 1)

        offline_ret = {}
        anchor_ret = {}
        anchor_ratio = {}
        for algorithm_name in eval_algorithms:
            offline_ret_ = {}
            anchor_ret_ = {}
            anchor_ratio_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(
                    env_settings().results_path, algorithm_name
                )
                base_results_path = "{}/{}".format(results_dir, seq.name)
                offline_path = "{}_offline.pkl".format(base_results_path)
                with open(offline_path, "rb") as fp:
                    offline_bb = pickle.load(fp)
                offline_bb.insert(0, None)

                results = []
                for box in offline_bb:
                    if box is not None:
                        results += box.tolist()
                results = np.array(results)
                if len(results) > 0:
                    results_center = self.convert_bb_to_center(results)
                    offline_gt = gt_traj[: len(results)]
                    offline_gt_center = self.convert_bb_to_center(offline_gt)
                    offline_ret_[seq.name] = success_error(
                        offline_gt_center, results_center, thresholds, len(results)
                    )
                else:
                    offline_ret_[seq.name] = np.array([0])

                valid_idx = [x is not None for x in offline_bb]
                anchor_box = np.array(
                    [offline_bb[i][-1] for i in range(len(offline_bb)) if valid_idx[i]]
                )
                if len(anchor_box) > 0:
                    anchor_box_center = self.convert_bb_to_center(anchor_box)
                    anchor_gt = np.array(
                        [gt_traj[i] for i in range(len(gt_traj)) if valid_idx[i]]
                    )
                    anchor_gt_center = self.convert_bb_to_center(anchor_gt)
                    anchor_ret_[seq.name] = np.sqrt(
                        np.sum(
                            np.power(anchor_gt_center - anchor_box_center, 2), axis=1
                        )
                    )
                else:
                    anchor_ret_[seq.name] = np.array([0])

                anchor_ratio_[seq.name] = sum(valid_idx) / len(gt_traj)
            offline_ret[algorithm_name] = offline_ret_
            anchor_ret[algorithm_name] = anchor_ret_
            anchor_ratio[algorithm_name] = anchor_ratio_
        return offline_ret, anchor_ret, anchor_ratio

    def eval_norm_precision(self, eval_trackers):
        """
        Args:
            eval_trackers: list of tracker
        Return:
            res: dict of results
        """

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for seq in self.dataset:
                gt_traj = np.array(seq.ground_truth_rect)
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
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

    def show_result(
        self,
        success_ret,
        precision_ret=None,
        show_video_level=False,
        helight_threshold=0.6,
    ):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc

        tracker_dp = {}
        for tracker_name in precision_ret.keys():
            dp = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            tracker_dp[tracker_name] = dp

        # sort tracker
        tracker_auc_ = sorted(tracker_auc.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]
        tracker_names = [x[0] for x in tracker_auc_]

        tracker_name_len = max((max([len(x) for x in success_ret.keys()]) + 2), 12)
        header = ("|{:^" + str(tracker_name_len) + "}|{:^9}|{:^11}|").format(
            "Tracker name", "Success", "Precision"
        )
        formatter = "|{:^" + str(tracker_name_len) + "}|{:^9.3f}|{:^11.3f}|"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for tracker_name in tracker_names:
            success = tracker_auc[tracker_name]
            precision = tracker_dp[tracker_name]
            print(formatter.format(tracker_name, success, precision))
        print("-" * len(header))

        if show_video_level and precision_ret is not None:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print("-" * len(header1))
            print(header1)
            print("-" * len(header1))
            print(header2)
            print("-" * len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f"{Fore.RED}{success_str}{Style.RESET_ALL}|"
                    else:
                        row += success_str + "|"
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f"{Fore.RED}{precision_str}{Style.RESET_ALL}|"
                    else:
                        row += precision_str + "|"
                print(row)
            print("-" * len(header1))

    def show_result_offline(
        self, success_ret, overlap_ret, precision_ret, dist_ret, anchor_ratio
    ):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc

        tracker_overlap = {}
        for tracker_name in overlap_ret.keys():
            values = []
            for value in overlap_ret[tracker_name].values():
                values += value.tolist()
            overlap = np.mean(values)
            tracker_overlap[tracker_name] = overlap

        tracker_dp = {}
        for tracker_name in precision_ret.keys():
            dp = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            tracker_dp[tracker_name] = dp

        tracker_dist = {}
        for tracker_name in dist_ret.keys():
            values = []
            for value in dist_ret[tracker_name].values():
                values += value.tolist()
            dist = np.mean(values)
            tracker_dist[tracker_name] = 1 / dist

        tracker_ratio = {}
        for tracker_name in anchor_ratio.keys():
            ratio = np.mean(list(anchor_ratio[tracker_name].values()))
            tracker_ratio[tracker_name] = ratio

        # sort tracker
        tracker_auc_ = sorted(tracker_auc.items(), key=lambda x: x[1], reverse=True)
        tracker_names = [x[0] for x in tracker_auc_]

        tracker_name_len = max((max([len(x) for x in success_ret.keys()]) + 2), 12)
        header = (
            "|{:^" + str(tracker_name_len) + "}|{:^9}|{:^11}|{:^9}|{:^6}|{:^14}|"
        ).format("Algorithm", "Success", "Precision", "Overlap", "Dist", "Anchor ratio")
        formatter = (
            "|{:^"
            + str(tracker_name_len)
            + "}|{:^9.3f}|{:^11.3f}|{:^9.3f}|{:^6.3f}|{:^14.3f}|"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for tracker_name in tracker_names:
            success = tracker_auc[tracker_name]
            overlap = tracker_overlap[tracker_name]
            precision = tracker_dp[tracker_name]
            dist = tracker_dist[tracker_name]
            ratio = tracker_ratio[tracker_name]
            print(
                formatter.format(tracker_name, success, precision, overlap, dist, ratio)
            )
        print("-" * len(header))
