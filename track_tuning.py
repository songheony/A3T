import numpy as np
from datasets.got10kdataset import GOT10KDatasetVal
from evaluations.ope_benchmark import OPEBenchmark
from track_dataset import run_tracker
from options import select_algorithms


def main(algorithm_name, experts, thresholds, **kwargs):
    algorithms = []
    dataset = GOT10KDatasetVal()

    for threshold in thresholds:
        kwargs["iou_threshold"] = threshold
        algorithm = select_algorithms(algorithm_name, experts, **kwargs)

        run_tracker(algorithm, dataset, experts=experts)
        algorithms.append(algorithm.name)

    benchmark = OPEBenchmark(dataset)

    success = benchmark.eval_success(experts)
    precision = benchmark.eval_precision(experts)
    benchmark.show_result(success, precision, show_video_level=False)

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

    start_point = 0.6
    end_point = 0.9
    thresholds = np.arange(start_point, end_point, 0.01)

    main(
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
