import os
from pathlib import Path
import numpy as np
import path_config
from datasets.data import SequenceList
from datasets.otbdataset import OTBDatasetClass


def OTBNoisyDataset(frame_ratio=1.0):
    return OTBNoisyDatasetClass(frame_ratio).get_sequence_list()


class OTBNoisyDatasetClass(OTBDatasetClass):
    """ OTB-2015 dataset with noise

    """

    def __init__(self, frame_ratio):
        super().__init__()

        self.frame_ratio = frame_ratio
        self.idx_dir = Path(path_config.OTB_NOISY_PATH) / str(self.frame_ratio)
        os.makedirs(self.idx_dir, exist_ok=True)

    def get_sequence_list(self):
        return SequenceList([self.noisy_sequence(s) for s in self.sequence_info_list])

    def noisy_sequence(self, sequence_info):
        seq = self._construct_sequence(sequence_info)

        idx_path = self.idx_dir / f"{seq.name}.txt"
        if idx_path.exists():
            idxs = np.loadtxt(idx_path, dtype=int)
        else:
            n_frames = len(seq.frames) - 1
            selected_frames = int(n_frames * self.frame_ratio)
            idxs = np.sort(np.random.choice(n_frames, selected_frames, replace=False))

            # add 0
            idxs += 1
            idxs = np.insert(idxs, 0, 0)

            idxs = idxs.astype(int)
            np.savetxt(idx_path, idxs, fmt="%i")

        seq.frames = [seq.frames[idx] for idx in idxs]
        seq.ground_truth_rect = seq.ground_truth_rect[idxs]
        seq.name = f"{seq.name}_{self.frame_ratio}"

        return seq
