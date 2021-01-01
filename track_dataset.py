import os
import pickle
import multiprocessing
from itertools import product
import random
import numpy as np
import torch
from select_options import select_datasets

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)


def run_sequence(dataset_name, seq, tracker, experts=None, debug=False):
    """Runs a tracker on a sequence."""

    dataset_path = "{}/{}".format(tracker.results_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    base_results_path = "{}/{}".format(dataset_path, seq.name)
    results_path = "{}.txt".format(base_results_path)
    times_path = "{}_time.txt".format(base_results_path)
    weights_path = "{}_weight.txt".format(base_results_path)
    offline_path = "{}_offline.pkl".format(base_results_path)

    if not debug and os.path.isfile(results_path):
        return

    print("Tracker: {},  Sequence: {}".format(tracker.name, seq.name))

    if debug:
        tracked_bb, offline_bb, weights, exec_times = tracker.run(
            dataset_name, seq, experts
        )
    else:
        try:
            tracked_bb, offline_bb, weights, exec_times = tracker.run(
                dataset_name, seq, experts
            )
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


def run_dataset(dataset, dataset_name, trackers, experts=None, threads=0, debug=False):
    """Runs a list of experts on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        experts: List of Tracker instances.
        debug: Debug level.
    """
    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(dataset_name, seq, tracker_info, experts=experts, debug=debug)
    else:
        param_list = [(dataset_name, seq, tracker_info, experts, debug) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')


def run(tracker, dataset_name, experts=None):
    dataset = select_datasets(dataset_name)

    run_dataset(dataset, dataset_name, [tracker], experts=experts, debug=False)
