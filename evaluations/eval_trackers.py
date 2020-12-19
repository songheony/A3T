import os
import pickle
from pathlib import Path
from path_config import EVALUATION_PATH
from evaluations.ope_benchmark import OPEBenchmark


def save_pickle(dir_path, filename, func, *args):
    file_path = dir_path / f"{filename}.pkl"

    if file_path.exists():
        data = pickle.loads(file_path.read_bytes())
    else:
        data = func(*args)
        file_path.write_bytes(pickle.dumps(data))

    return data


def evaluate(datasets, datasets_name, experts, baselines, algorithm, save_dir=None):
    if save_dir is None:
        save_dir = Path(EVALUATION_PATH)
    os.makedirs(save_dir, exist_ok=True)

    if algorithm is not None:
        eval_trackers = experts + baselines + [algorithm]
    else:
        eval_trackers = experts

    tracking_time_rets = {}
    success_rets = {}
    precision_rets = {}
    norm_precision_rets = {}
    anchor_success_rets = {}
    anchor_precision_rets = {}
    anchor_norm_precision_rets = {}
    error_rets = {}
    loss_rets = {}
    anchor_frame_rets = {}

    for dataset, dataset_name in zip(datasets, datasets_name):
        ope = OPEBenchmark(dataset)

        tracking_time_rets[dataset_name] = {}
        success_rets[dataset_name] = {}
        precision_rets[dataset_name] = {}
        norm_precision_rets[dataset_name] = {}
        anchor_success_rets[dataset_name] = {}
        anchor_precision_rets[dataset_name] = {}
        anchor_norm_precision_rets[dataset_name] = {}
        error_rets[dataset_name] = {}
        loss_rets[dataset_name] = {}

        if algorithm is not None:
            anchor_frame_rets[dataset_name] = ope.get_anchor_frames(dataset_name, algorithm)

        for tracker_name in eval_trackers:
            tracker_dir = save_dir / tracker_name / dataset_name
            os.makedirs(tracker_dir, exist_ok=True)

            tracking_time = save_pickle(tracker_dir, "tracking_time", ope.eval_times, dataset_name, tracker_name)
            tracking_time_rets[dataset_name][tracker_name] = tracking_time

            success = save_pickle(tracker_dir, "success", ope.eval_success, dataset_name, tracker_name)
            success_rets[dataset_name][tracker_name] = success

            precision = save_pickle(tracker_dir, "precision", ope.eval_precision, dataset_name, tracker_name)
            precision_rets[dataset_name][tracker_name] = precision

            norm_precision = save_pickle(tracker_dir, "norm_precision", ope.eval_norm_precision, dataset_name, tracker_name)
            norm_precision_rets[dataset_name][tracker_name] = norm_precision

            if algorithm is not None:
                anchor_success = save_pickle(tracker_dir, "anchor_success", ope.eval_success, dataset_name, tracker_name, anchor_frame_rets[dataset_name])
                anchor_success_rets[dataset_name][tracker_name] = anchor_success

                anchor_precision = save_pickle(tracker_dir, "anchor_precision", ope.eval_precision, dataset_name, tracker_name, anchor_frame_rets[dataset_name])
                anchor_precision_rets[dataset_name][tracker_name] = anchor_precision

                anchor_norm_precision = save_pickle(tracker_dir, "anchor_norm_precision", ope.eval_norm_precision, dataset_name, tracker_name, anchor_frame_rets[dataset_name])
                anchor_norm_precision_rets[dataset_name][tracker_name] = anchor_norm_precision

                if tracker_name != algorithm:
                    error, loss = save_pickle(tracker_dir, "loss", ope.eval_loss, dataset_name, tracker_name)
                    error_rets[dataset_name][tracker_name] = error
                    loss_rets[dataset_name][tracker_name] = loss

    return tracking_time_rets, success_rets, precision_rets, norm_precision_rets, anchor_success_rets, anchor_precision_rets, anchor_norm_precision_rets, error_rets, loss_rets, anchor_frame_rets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default=None, type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-b", "--baselines", default=list(), nargs="+")
    args = parser.parse_args()

    eval_dir = Path("./evaluation_results")
    evaluate(args.experts, args.baselines, args.algorithm, eval_dir)
