import os
import numpy as np
import pandas as pd
import path_config
from datasets.data import Sequence, BaseDataset, SequenceList


def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except Exception:
                pass

        raise Exception("Could not read file {}".format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(
                    path,
                    delimiter=d,
                    header=None,
                    dtype=dtype,
                    na_filter=False,
                    low_memory=False,
                ).values
                return ground_truth_rect
            except Exception:
                pass

        raise Exception("Could not read file {}".format(path))
    else:
        ground_truth_rect = pd.read_csv(
            path,
            delimiter=delimiter,
            header=None,
            dtype=dtype,
            na_filter=False,
            low_memory=False,
        ).values
        return ground_truth_rect


def load_text(path, delimiter=" ", dtype=np.float32, backend="numpy"):
    if backend == "numpy":
        return load_text_numpy(path, delimiter, dtype)
    elif backend == "pandas":
        return load_text_pandas(path, delimiter, dtype)


def TrackingNetDataset():
    return TrackingNetClass().get_sequence_list()


class TrackingNetClass(BaseDataset):
    """ TrackingNet test set.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """

    def __init__(self):
        super().__init__()
        self.base_path = path_config.TRACKINGNET_PATH

        sets = "TEST"
        if not isinstance(sets, (list, tuple)):
            if sets == "TEST":
                sets = ["TEST"]
            elif sets == "TRAIN":
                sets = ["TRAIN_{}".format(i) for i in range(5)]

        self.sequence_list = self._list_sequences(self.base_path, sets)

    def get_sequence_list(self):
        return SequenceList(
            [
                self._construct_sequence(set, seq_name)
                for set, seq_name in self.sequence_list
            ]
        )

    def _construct_sequence(self, set, sequence_name):
        anno_path = "{}/{}/anno/{}.txt".format(self.base_path, set, sequence_name)

        ground_truth_rect = load_text(
            str(anno_path), delimiter=",", dtype=np.float64, backend="numpy"
        )

        frames_path = "{}/{}/frames/{}".format(self.base_path, set, sequence_name)
        frame_list = [
            frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")
        ]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(
            sequence_name, frames_list, "trackingnet", ground_truth_rect.reshape(-1, 4)
        )

    def __len__(self):
        return len(self.sequence_list)

    def _list_sequences(self, root, set_ids):
        sequence_list = []

        for s in set_ids:
            anno_dir = os.path.join(root, s, "anno")
            sequences_cur_set = [
                (s, os.path.splitext(f)[0])
                for f in os.listdir(anno_dir)
                if f.endswith(".txt")
            ]

            sequence_list += sequences_cur_set

        return sequence_list
