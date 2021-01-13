import os
from pathlib import Path
import numpy as np
from track_dataset import run_dataset
from select_options import select_algorithms, select_datasets
from evaluations.eval_trackers import evaluate
from visualizes.draw_tables import get_mean_succ, get_mean_prec
import path_config


def main(algorithm_name, experts, save_dir, mode):
    algorithms = []
    dataset_name = "Got10K"
    dataset = select_datasets(dataset_name)

    if algorithm_name == "AAA":
        thresholds = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]
        feature_factors = [1, 3, 5, 7, 9, 11, 13]
    elif algorithm_name == "HDT":
        thresholds = np.linspace(0.0, 1.0, 11)
    elif algorithm_name == "MCCT":
        thresholds = np.linspace(0.0, 1.0, 11)

    for threshold in thresholds:
        for feature_factor in feature_factors:
            algorithm = select_algorithms(
                algorithm_name,
                experts,
                mode=mode,
                threshold=threshold,
                feature_factor=feature_factor,
            )
            run_dataset(
                dataset, dataset_name, [algorithm], experts=experts, threads=8
            )
            algorithms.append(algorithm.name)

            (
                tracking_time_rets,
                success_rets,
                precision_rets,
                norm_precision_rets,
                anchor_success_rets,
                anchor_precision_rets,
                anchor_norm_precision_rets,
                error_rets,
                loss_rets,
                anchor_frame_rets,
            ) = evaluate(
                [dataset], [dataset_name], [], [], algorithm.name, save_dir
            )

            auc = get_mean_succ([algorithm.name], [dataset_name], success_rets)[
                0
            ][0]
            prec = get_mean_prec(
                [algorithm.name], [dataset_name], precision_rets
            )[0][0]
            print(f"AUC: {auc}, DP: {prec}")


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
