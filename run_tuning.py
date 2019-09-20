import numpy as np
import multiprocessing
import os
import pickle
from itertools import product
from datasets.got10kdataset import GOT10KDatasetVal
from evaluations.ope_benchmark import OPEBenchmark


def run_sequence(seq, algorithm, experts, debug=False):
    """Runs a tracker on a sequence."""

    base_results_path = "{}/{}".format(algorithm.results_dir, seq.name)
    results_path = "{}.txt".format(base_results_path)
    times_path = "{}_time.txt".format(base_results_path)
    weights_path = "{}_weight.txt".format(base_results_path)
    offline_path = "{}_offline.pkl".format(base_results_path)

    if os.path.isfile(results_path):
        return

    print("Tracker: {},  Sequence: {}".format(algorithm.name, seq.name))

    if debug:
        tracked_bb, offline_bb, weights, exec_times = algorithm.run(
            seq, experts, input_gt=False
        )
    else:
        try:
            tracked_bb, offline_bb, weights, exec_times = algorithm.run(
                seq, experts, input_gt=False
            )
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(float)
    exec_times = np.array(exec_times).astype(float)

    print(
        "FPS: {} Anchor: {}".format(
            len(exec_times) / exec_times.sum(),
            (sum(x is not None for x in offline_bb) + 1) / len(tracked_bb),
        )
    )
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter="\t", fmt="%f")
        np.savetxt(weights_path, weights, delimiter="\t", fmt="%f")
        np.savetxt(times_path, exec_times, delimiter="\t", fmt="%f")
        with open(offline_path, "wb") as fp:
            pickle.dump(offline_bb, fp)


def run_dataset(dataset, algorithms, experts, debug=False, threads=0):
    """Runs a list of experts on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        experts: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    if threads == 0:
        mode = "sequential"
    else:
        mode = "parallel"

    if mode == "sequential":
        for seq in dataset:
            for algorithm_info in algorithms:
                run_sequence(seq, algorithm_info, experts, debug=debug)
    elif mode == "parallel":
        param_list = [
            (seq, algorithm_info, experts, debug)
            for seq, algorithm_info in product(dataset, algorithms)
        ]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print("Done")


def run_tracker(algorithm, experts, dataset, sequence=None, debug=0, threads=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset: Dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    if sequence is not None:
        dataset = [dataset[sequence]]

    algorithms = [algorithm]

    run_dataset(dataset, algorithms, experts, debug, threads)


def main(algorithm_name, experts, thresholds):
    algorithms = []
    n_experts = len(experts)
    dataset = GOT10KDatasetVal()

    for threshold in thresholds:
        if algorithm_name == "AAA":
            from algorithms.aaa import AAA

            algorithm = AAA(
                n_experts,
                iou_threshold=0.0,
                feature_threshold=threshold,
                reset_target=False,
                only_max=False,
                use_iou=False,
                use_feature=True,
                cost_iou=True,
                cost_feature=True,
                cost_score=True,
            )
        elif algorithm_name == "AAA_select":
            from algorithms.aaa_select import AAA_select

            algorithm = AAA_select(
                n_experts,
                iou_threshold=0.0,
                feature_threshold=threshold,
                reset_target=True,
                only_max=False,
                use_iou=False,
                use_feature=True,
                cost_iou=True,
                cost_feature=True,
                cost_score=True,
            )
        else:
            raise ValueError("Unknown algorithm name")

        run_tracker(algorithm, experts, dataset, debug=0)
        algorithms.append(algorithm.name)

    benchmark = OPEBenchmark(dataset)

    # success = benchmark.eval_success(experts)
    # precision = benchmark.eval_precision(experts)
    # benchmark.show_result(success, precision, show_video_level=False)

    success = benchmark.eval_success(algorithms)
    precision = benchmark.eval_precision(algorithms)
    benchmark.show_result(success, precision, show_video_level=False)

    success_offline, overlap_anchor, anchor_ratio = benchmark.eval_success_offline(
        algorithms
    )
    precision_offline, dist_anchor, anchor_ratio = benchmark.eval_precision_offline(
        algorithms
    )
    benchmark.show_result_offline(
        success_offline, overlap_anchor, precision_offline, dist_anchor, anchor_ratio
    )


if __name__ == "__main__":
    import argparse

    experts = ["ATOM", "DaSiamRPN", "ECO", "SiamDW", "SiamRPN++", "TADT"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA_select", type=str)
    parser.add_argument("-e", "--experts", default=experts, nargs="+")
    parser.add_argument("-s", "--start_point", default=0.7, type=float)
    parser.add_argument("-t", "--end_point", default=0.9, type=float)
    parser.add_argument("-n", "--sample_number", default=21, type=int)
    args = parser.parse_args()

    thresholds = np.linspace(args.start_point, args.end_point, num=args.sample_number)

    main(args.algorithm, args.experts, thresholds)
