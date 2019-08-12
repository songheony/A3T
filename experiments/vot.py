import os

from .otb import ExperimentOTB
from external.toolkit.got10k.datasets import VOT


class ExperimentVOT(ExperimentOTB):
    r"""Experiment pipeline and evaluation toolkit for VOT dataset.

    Notes:
        - The original VOT toolkit provides three types of experiments ``supervised``
            ``unsupervised`` and ``realtime``. However in this code, we only use unsupervised experiment.

    Args:
        root_dir (string): Root directory of VOT dataset where sequence
            folders exist.
        version (integer, optional): Specify the VOT dataset version. Specify as
            one of 2013~2018. Default is 2017.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(
        self, root_dir, version=2017, result_dir="results", report_dir="reports"
    ):
        self.dataset = VOT(
            root_dir,
            version,
            anno_type="rect",
            download=True,
            return_meta=True,
            list_file=None,
        )
        if version == "LT2018":
            version = "-" + version
        self.result_dir = os.path.join(result_dir, "VOT" + str(version))
        self.report_dir = os.path.join(report_dir, "VOT" + str(version))
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.tags = [
            "camera_motion",
            "illum_change",
            "occlusion",
            "size_change",
            "motion_change",
            "empty",
        ]

    def run(self, tracker, visualize=False):
        print(
            "Running tracker %s on %s..." % (tracker.name, type(self.dataset).__name__)
        )

        # loop over the complete dataset
        for s, (img_files, anno, _) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print("--Sequence %d/%d: %s" % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, "%s.txt" % seq_name
            )
            if os.path.exists(record_file):
                print("  Found results, skipping", seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            self._record(record_file, boxes, times)
