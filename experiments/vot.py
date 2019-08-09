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
            anno_type="default",
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
