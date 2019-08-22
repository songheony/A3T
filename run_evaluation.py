from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from evaluations.ope_benchmark import OPEBenchmark


def main(trackers, algorithms, dataset_name):
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

    benchmark = OPEBenchmark(dataset)
    success = benchmark.eval_success(trackers)
    precision = benchmark.eval_precision(trackers)
    benchmark.show_result(success, precision)

    success_anchor, anchor_ratio = benchmark.eval_success_anchor(algorithms)
    precision_anchor = benchmark.eval_precision_anchor(algorithms)
    benchmark.show_result_anchor(
        success_anchor, precision_anchor, anchor_ratio=anchor_ratio
    )


if __name__ == "__main__":
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
        "Average",
        "MCCT",
    ]

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    baselines = (
        ["Overlap_%s" % threshold for threshold in thresholds]
        + ["Similar_%s" % threshold for threshold in thresholds]
        + ["Both_%s" % threshold for threshold in thresholds]
    )

    dataset = "OTB"

    main(trackers, baselines, dataset)
