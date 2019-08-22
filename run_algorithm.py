import numpy as np
import multiprocessing
import os
import pickle
from itertools import product
from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset


def run_sequence(seq, algorithm, trackers, debug=False):
    """Runs a tracker on a sequence."""

    base_results_path = "{}/{}".format(algorithm.results_dir, seq.name)
    results_path = "{}.txt".format(base_results_path)
    times_path = "{}_time.txt".format(base_results_path)
    weights_path = "{}_weight.txt".format(base_results_path)
    offline_path = "{}_offline.pkl".format(base_results_path)

    if os.path.isfile(results_path) and not debug:
        return

    print("Tracker: {},  Sequence: {}".format(algorithm.name, seq.name))

    if debug:
        tracked_bb, offline_bb, weights, exec_times = algorithm.run(seq, trackers)
    else:
        try:
            tracked_bb, offline_bb, weights, exec_times = algorithm.run(seq, trackers)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(float)
    exec_times = np.array(exec_times).astype(float)

    print(
        "FPS: {} Anchor: {}".format(
            len(exec_times) / exec_times.sum(),
            sum(x is not None for x in offline_bb) / len(offline_bb),
        )
    )
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter="\t", fmt="%f")
        np.savetxt(weights_path, weights, delimiter="\t", fmt="%f")
        np.savetxt(times_path, exec_times, delimiter="\t", fmt="%f")
        with open(offline_path, "wb") as fp:
            pickle.dump(offline_bb, fp)


def run_dataset(dataset, algorithms, trackers, debug=False, threads=0):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
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
                run_sequence(seq, algorithm_info, trackers, debug=debug)
    elif mode == "parallel":
        param_list = [
            (seq, algorithm_info, trackers, debug)
            for seq, algorithm_info in product(dataset, algorithms)
        ]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print("Done")


def run_tracker(algorithm, trackers, dataset, sequence=None, debug=0, threads=0):
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

    run_dataset(dataset, algorithms, trackers, debug, threads)


def main(algorithm_name, trackers, dataset_name, **kargs):
    n_experts = len(trackers)
    if algorithm_name == "AAA":
        from algorithms.aaa import AAA

        algorithm = AAA(n_experts, **kargs)
    elif algorithm_name == "Average":
        from algorithms.average import Average

        algorithm = Average(n_experts)
    elif algorithm_name == "Both":
        from algorithms.baseline import Baseline

        algorithm = Baseline(
            n_experts,
            name="Both_%s" % kargs["threshold"],
            threshold=kargs["threshold"],
            use_iou=True,
            use_feature=True,
        )
    elif algorithm_name == "MCCT":
        from algorithms.mcct import MCCT

        algorithm = MCCT(n_experts)
    elif algorithm_name == "Overlap":
        from algorithms.baseline import Baseline

        algorithm = Baseline(
            n_experts,
            name="Overlap_%s" % kargs["threshold"],
            threshold=kargs["threshold"],
            use_iou=True,
            use_feature=False,
        )
    elif algorithm_name == "Similar":
        from algorithms.baseline import Baseline

        algorithm = Baseline(
            n_experts,
            name="Similar_%s" % kargs["threshold"],
            threshold=kargs["threshold"],
            use_iou=False,
            use_feature=True,
        )
    else:
        raise ValueError("Unknown algorithm name")

    if dataset_name == "OTB":
        dataset = OTBDataset()
    elif dataset_name == "NFS":
        dataset = NFSDataset()
    elif dataset_name == "UAV":
        dataset = UAVDataset()
    elif dataset_name == "TPL":
        dataset = TPLDataset()
    elif dataset_name == "VOT":
        dataset = VOTDataset()
    elif dataset_name == "LaSOT":
        dataset = LaSOTDataset()
    else:
        raise ValueError("Unknown dataset name")

    run_tracker(algorithm, trackers, dataset, debug=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="Both", type=str)
    parser.add_argument("-e", "--experts", default=[], nargs="+")
    parser.add_argument("-d", "--dataset", default="OTB", type=str)
    parser.add_argument("-t", "--threshold", default=0.0, type=float)
    parser.add_argument("-m", "--only_max", action="store_true")
    parser.add_argument("-i", "--use_iou", action="store_true")
    parser.add_argument("-f", "--use_feature", action="store_true")
    parser.add_argument("-x", "--cost_iou", action="store_true")
    parser.add_argument("-y", "--cost_feature", action="store_true")
    parser.add_argument("-z", "--cost_score", action="store_true")
    args = parser.parse_args()

    if len(args.experts) == 0:
        trackers = [
            "ATOM",
            "DaSiamRPN",
            "ECO",
            "SiamDW",
            "SiamFC",
            "SiamRPN",
            "SiamRPN++",
            "Staple",
            "STRCF",
            "TADT",
        ]
    else:
        trackers = args.experts

    main(
        args.algorithm,
        trackers,
        args.dataset,
        threshold=args.threshold,
        only_max=args.only_max,
        use_iou=args.use_iou,
        use_feature=args.use_feature,
        cost_iou=args.cost_iou,
        cost_feature=args.cost_feature,
        cost_score=args.cost_score,
    )
