import os
from pathlib import Path
from track_dataset import run_dataset
from select_options import select_algorithms, select_datasets
import path_config


def main(algorithm_name, experts, save_dir, mode):
    algorithms = []
    dataset_name = "Got10K"
    dataset = select_datasets(dataset_name)

    if algorithm_name == "AAA":
        thresholds = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
    elif algorithm_name == "HDT":
        thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for threshold in thresholds:

        algorithm = select_algorithms(
            algorithm_name, experts, mode=mode, threshold=threshold,
        )
        run_dataset(dataset, dataset_name, [algorithm], experts=experts, threads=8)
        algorithms.append(algorithm.name)


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
