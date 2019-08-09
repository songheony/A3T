import os

from .otb import ExperimentOTB
from external.toolkit.got10k.datasets import LaSOT


class ExperimentLaSOT(ExperimentOTB):
    r"""Experiment pipeline and evaluation toolkit for LaSOT dataset.

    Args:
        root_dir (string): Root directory of LaSOT dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, result_dir="results", report_dir="reports"):
        self.dataset = LaSOT(root_dir)
        self.result_dir = os.path.join(result_dir, "LaSOT")
        self.report_dir = os.path.join(report_dir, "LaSOT")
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
