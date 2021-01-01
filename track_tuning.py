import os
from pathlib import Path
import numpy as np
from track_dataset import run_dataset
from select_options import select_algorithms, select_datasets
from evaluations.eval_trackers import evaluate
import path_config


def main(algorithm_name, experts, save_dir, mode):
    algorithms = []
    dataset_name = "Got10K"
    dataset = select_datasets(dataset_name)

    if "AAA" in args.algorithm:
        thresholds = np.linspace(0.9, 1.0, 20, endpoint=False)
        offline_bs = np.linspace(1, 5, 5, dtype=int)
        offline_cs = np.linspace(1, 5, 5, dtype=int)

        for threshold in thresholds:
            for offline_b in offline_bs:
                for offline_c in offline_cs:
                    algorithm = select_algorithms(algorithm_name, experts, mode=mode, threshold=threshold, offline_b=offline_b, offline_c=offline_c)
                    run_dataset(dataset, dataset_name, [algorithm], experts=experts, threads=8)
                    algorithms.append(algorithm.name)

                    evaluate([dataset], [dataset_name], [], [], algorithm.name, save_dir)

    else:
        thresholds = np.linspace(0.1, 1.0, 20, endpoint=False)
        for threshold in thresholds:
            algorithm = select_algorithms(algorithm_name, experts, mode=mode, threshold=threshold)

            run_tracker(algorithm, dataset, dataset_name, experts=experts)
            algorithms.append(algorithm.name)

            evaluate([dataset], [dataset_name], [], [], algorithm.name, save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument(
        "-e", "--experts", default=["DaSiamRPN", "SiamDW", "SiamRPN", "SPM"], nargs="+"
    )
    parser.add_argument("-m", "--mode", default="SuperFast", type=str)
    args = parser.parse_args()

    save_dir = Path(f"./{path_config.EVALUATION_PATH}")
    os.makedirs(save_dir, exist_ok=True)

    main(
        args.algorithm, args.experts, save_dir, args.mode,
    )
