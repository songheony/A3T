import os
from pathlib import Path
import numpy as np

from track_dataset import run_tracker
from select_options import select_algorithms, select_datasets
from evaluations.eval_trackers import evaluate
import path_config


def main(algorithm_name, experts, thresholds, save_dir, **kwargs):
    algorithms = []
    dataset_name = "Got10K"
    dataset = select_datasets(dataset_name)

    for threshold in thresholds:
        kwargs["threshold"] = threshold
        algorithm = select_algorithms(algorithm_name, experts, **kwargs)

        run_tracker(algorithm, dataset, dataset_name, experts=experts)
        algorithms.append(algorithm.name)

        evaluate([dataset], [dataset_name], [], [], algorithm.name, save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=["DaSiamRPN", "SiamDW", "SiamRPN", "SPM"], nargs="+")
    parser.add_argument("-n", "--mode", default="High", type=str)
    args = parser.parse_args()

    if "AAA" in args.algorithm:
        start_point = 0.5
        end_point = 1.0
        thresholds = np.arange(start_point, end_point, 0.01)
    else:
        start_point = 0.1
        end_point = 1.0
        thresholds = np.arange(start_point, end_point, 0.02)

    save_dir = Path(f"./{path_config.EVALUATION_PATH}/{args.algorithm}/{args.mode}")
    os.makedirs(save_dir, exist_ok=True)

    main(
        args.algorithm,
        args.experts,
        thresholds,
        save_dir,
        mode=args.mode,
    )
