import numpy as np
import multiprocessing
import os
from itertools import product
from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset


def run_sequence(seq, tracker, debug=False):
    """Runs a tracker on a sequence."""

    base_results_path = "{}/{}".format(tracker.results_dir, seq.name)
    results_path = "{}.txt".format(base_results_path)
    times_path = "{}_time.txt".format(base_results_path)

    if os.path.isfile(results_path) and not debug:
        return

    print("Tracker: {},  Sequence: {}".format(tracker.name, seq.name))

    if debug:
        tracked_bb, exec_times = tracker.run(seq)
    else:
        try:
            tracked_bb, exec_times = tracker.run(seq)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(int)
    exec_times = np.array(exec_times).astype(float)

    print("FPS: {}".format(len(exec_times) / exec_times.sum()))
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter="\t", fmt="%d")
        np.savetxt(times_path, exec_times, delimiter="\t", fmt="%f")


def run_dataset(dataset, trackers, debug=False, threads=0):
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
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == "parallel":
        param_list = [
            (seq, tracker_info, debug)
            for seq, tracker_info in product(dataset, trackers)
        ]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print("Done")


def run_tracker(tracker, dataset, sequence=None, debug=0, threads=0):
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

    trackers = [tracker]

    run_dataset(dataset, trackers, debug, threads)


if __name__ == "__main__":
    from experts.atom import ATOM

    tracker = ATOM()

    # from experts.dasiamrpn import DaSiamRPN

    # tracker = DaSiamRPN()

    # from experts.eco import ECO

    # tracker = ECO()

    # from experts.mdnet import MDnet

    # tracker = MDnet()

    # from experts.siamdw import SiamDW

    # tracker = SiamDW()

    # from experts.siamfc import SiamFC

    # tracker = SiamFC()

    # from experts.siamrpn import SiamRPN

    # tracker = SiamRPN()

    # from experts.tadt import TADT

    # tracker = TADT()

    # from experts.vital import Vital

    # tracker = Vital()

    # from experts.bacf import BACF

    # tracker = BACF()

    # from experts.csrdcf import CSRDCF

    # tracker = CSRDCF()

    # from experts.staple import Staple

    # tracker = Staple()

    # from experts.strcf import STRCF

    # tracker = STRCF()

    dataset_name = "otb"

    if dataset_name == "otb":
        dataset = OTBDataset()
    elif dataset_name == "nfs":
        dataset = NFSDataset()
    elif dataset_name == "uav":
        dataset = UAVDataset()
    elif dataset_name == "tpl":
        dataset = TPLDataset()
    elif dataset_name == "vot":
        dataset = VOTDataset()
    elif dataset_name == "lasot":
        dataset = LaSOTDataset()
    else:
        raise ValueError("Unknown dataset name")

    run_tracker(tracker, dataset)
