import numpy as np
import multiprocessing
import os
import pickle
from itertools import product
from options import select_datasets


def run_sequence(seq, tracker, experts=None, debug=False):
    """Runs a tracker on a sequence."""

    base_results_path = "{}/{}".format(tracker.results_dir, seq.name)
    results_path = "{}.txt".format(base_results_path)
    times_path = "{}_time.txt".format(base_results_path)
    weights_path = "{}_weight.txt".format(base_results_path)
    offline_path = "{}_offline.pkl".format(base_results_path)

    if not debug and os.path.isfile(results_path):
        return

    print("Tracker: {},  Sequence: {}".format(tracker.name, seq.name))

    if debug:
        tracked_bb, offline_bb, weights, exec_times = tracker.run(seq, experts)
    else:
        try:
            tracked_bb, offline_bb, weights, exec_times = tracker.run(seq, experts)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(float)
    exec_times = np.array(exec_times).astype(float)

    if experts is not None:
        print(
            "FPS: {} Anchor: {}".format(
                len(exec_times) / exec_times.sum(),
                (sum(x is not None for x in offline_bb) + 1) / len(tracked_bb),
            )
        )
    else:
        print("FPS: {}".format(len(exec_times) / exec_times.sum()))
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter="\t", fmt="%f")
        np.savetxt(times_path, exec_times, delimiter="\t", fmt="%f")
        if experts is not None:
            np.savetxt(weights_path, weights, delimiter="\t", fmt="%f")
            with open(offline_path, "wb") as fp:
                pickle.dump(offline_bb, fp)


def run_dataset(dataset, trackers, experts=None, debug=False, threads=0):
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
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, experts=experts, debug=debug)
    elif mode == "parallel":
        param_list = [
            (seq, tracker_info, experts, debug)
            for seq, tracker_info in product(dataset, trackers)
        ]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print("Done")


def run_tracker(tracker, dataset, experts=None, sequence=None, debug=0, threads=0):
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

    run_dataset(dataset, [tracker], experts=experts, debug=debug, threads=threads)


def run(tracker, dataset_name, experts=None):
    dataset = select_datasets(dataset_name)

    run_tracker(tracker, dataset, experts=experts, debug=False)
