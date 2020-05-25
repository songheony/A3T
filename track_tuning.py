import os
import pickle
from pathlib import Path
import numpy as np

from track_dataset import run_tracker
from options import select_algorithms
from datasets.got10kdataset import GOT10KDatasetVal
from evaluations.ope_benchmark import OPEBenchmark
from evaluations.offline_benchmark import OfflineBenchmark
from visualize_eval import make_ratio_table, draw_score_with_ratio


def vis_all(algorithm_name, eval_dir):
    dataset_name = "GOT10K"
    modes = ["Good", "Bad", "Mix", "SiamDW", "SiamRPN++"]
    mode_names = ["High", "Low", "Mix", "SiamDW", "SiamRPN++"]

    threshold_successes = {mode: {} for mode in modes}
    threshold_precisions = {mode: {} for mode in modes}
    threshold_anchor_successes = {mode: {} for mode in modes}
    threshold_anchor_precisions = {mode: {} for mode in modes}
    threshold_offline_successes = {mode: {} for mode in modes}
    threshold_offline_precisions = {mode: {} for mode in modes}
    threshold_anchors = {mode: {} for mode in modes}

    for mode in modes:
        eval_save = eval_dir / mode / "eval.pkl"
        successes, precisions, anchor_frames, anchor_successes, anchor_precisions, offline_successes, offline_precisions = pickle.loads(
            eval_save.read_bytes()
        )

        for algorithm in successes[dataset_name].keys():
            if algorithm_name.startswith("AAA"):
                algorithm_split_name = algorithm.split("_")[3]
            elif algorithm_name.startswith("MCCT"):
                algorithm_split_name = algorithm.split("_")[2]

            threshold_successes[mode][algorithm_split_name] = successes[dataset_name][
                algorithm
            ]
            threshold_precisions[mode][algorithm_split_name] = precisions[dataset_name][
                algorithm
            ]
            if algorithm_name.startswith("AAA"):
                threshold_anchor_successes[mode][
                    algorithm_split_name
                ] = anchor_successes[dataset_name][algorithm]
                threshold_anchor_precisions[mode][
                    algorithm_split_name
                ] = anchor_precisions[dataset_name][algorithm]
                threshold_offline_successes[mode][
                    algorithm_split_name
                ] = offline_successes[dataset_name][algorithm]
                threshold_offline_precisions[mode][
                    algorithm_split_name
                ] = offline_precisions[dataset_name][algorithm]
                threshold_anchors[mode][algorithm_split_name] = anchor_frames[
                    dataset_name
                ][algorithm]

    figsize = (20, 5)

    names = sorted(threshold_successes[mode].keys())
    if algorithm_name.startswith("AAA"):
        make_ratio_table(
            modes, names, threshold_successes, threshold_anchors, eval_dir, "score"
        )
        draw_score_with_ratio(
            modes,
            mode_names,
            names,
            threshold_successes,
            threshold_anchors,
            figsize,
            eval_dir,
        )
        make_ratio_table(
            modes,
            names,
            threshold_anchor_successes,
            threshold_anchors,
            eval_dir,
            "anchor",
        )

        make_ratio_table(
            modes,
            names,
            threshold_offline_successes,
            threshold_anchors,
            eval_dir,
            "offline",
        )
    elif algorithm_name.startswith("MCCT"):
        make_ratio_table(modes, names, threshold_successes, None, eval_dir, "score")
        draw_score_with_ratio(
            modes, names, threshold_successes, None, figsize, eval_dir
        )


def main(eval_dir, algorithm_name, experts, thresholds, **kwargs):
    algorithms = []
    dataset = GOT10KDatasetVal()
    dataset_name = "GOT10K"

    for threshold in thresholds:
        kwargs["feature_threshold"] = threshold
        algorithm = select_algorithms(algorithm_name, experts, **kwargs)

        run_tracker(algorithm, dataset, experts=experts)
        algorithms.append(algorithm.name)

    eval_save = eval_dir / "eval.pkl"
    if eval_save.exists():
        successes, precisions, anchor_frames, anchor_successes, anchor_precisions, offline_successes, offline_precisions = pickle.loads(
            eval_save.read_bytes()
        )
    else:
        # algorithms' performance
        ope = OPEBenchmark(dataset)
        successes = {dataset_name: ope.eval_success(algorithms)}
        precisions = {dataset_name: ope.eval_precision(algorithms)}

        # offline trackers' performance
        if algorithm_name.startswith("AAA"):
            offline = OfflineBenchmark(dataset)
            anchor_frames = {dataset_name: {}}
            anchor_successes = {dataset_name: {}}
            anchor_precisions = {dataset_name: {}}
            offline_successes = {dataset_name: {}}
            offline_precisions = {dataset_name: {}}
            for algorithm in algorithms:
                anchor_frame, anchor_success, anchor_precision = offline.eval_anchor_frame(
                    algorithm, experts
                )
                offline_success, offline_precision = offline.eval_offline_tracker(
                    algorithm, experts
                )

                anchor_frames[dataset_name][algorithm] = anchor_frame
                anchor_successes[dataset_name][algorithm] = anchor_success[algorithm]
                anchor_precisions[dataset_name][algorithm] = anchor_precision[algorithm]
                offline_successes[dataset_name][algorithm] = offline_success[algorithm]
                offline_precisions[dataset_name][algorithm] = offline_precision[
                    algorithm
                ]
        else:
            anchor_frames = None
            anchor_successes = None
            anchor_precisions = None
            offline_successes = None
            offline_precisions = None

        eval_save.write_bytes(
            pickle.dumps(
                (
                    successes,
                    precisions,
                    anchor_frames,
                    anchor_successes,
                    anchor_precisions,
                    offline_successes,
                    offline_precisions,
                )
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-n", "--mode", default="Expert", type=str)
    parser.add_argument("-s", "--reset_target", action="store_true")
    parser.add_argument("-m", "--only_max", action="store_true")
    parser.add_argument("-i", "--use_iou", action="store_true")
    parser.add_argument("-f", "--use_feature", action="store_false")
    parser.add_argument("-x", "--cost_iou", action="store_false")
    parser.add_argument("-y", "--cost_feature", action="store_false")
    parser.add_argument("-z", "--cost_score", action="store_false")
    args = parser.parse_args()

    vis_all("AAA", Path(f"./tuning_results/AAA/"))
    exit()

    if args.algorithm.startswith("AAA"):
        start_point = 0.6
        end_point = 0.9
        thresholds = np.arange(start_point, end_point, 0.01)
    elif args.algorithm.startswith("MCCT"):
        # start_point = 0.01
        # end_point = 0.11
        # thresholds = np.arange(start_point, end_point, 0.01)
        thresholds = [0.0001, 0.001, 0.01, 0.1, 1.0]

    eval_dir = Path(f"./tuning_results/{args.algorithm}/{args.mode}")
    os.makedirs(eval_dir, exist_ok=True)

    main(
        eval_dir,
        args.algorithm,
        args.experts,
        thresholds,
        mode=args.mode,
        reset_target=args.reset_target,
        only_max=args.only_max,
        use_iou=args.use_iou,
        use_feature=args.use_feature,
        cost_iou=args.cost_iou,
        cost_feature=args.cost_feature,
        cost_score=args.cost_score,
    )
