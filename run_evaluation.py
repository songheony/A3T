from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from evaluations.ope_benchmark import OPEBenchmark


def main(trackers, dataset_name):
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

    benchmark = OPEBenchmark(dataset)
    success = benchmark.eval_success(trackers)
    precision = benchmark.eval_precision(trackers)
    benchmark.show_result(success, precision)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experts", default="None", type=str, help="experts")
    parser.add_argument("-d", "--dataset", default="OTB", type=str, help="dataset")
    args = parser.parse_args()

    if args.experts == "None":
        trackers = [
            "ATOM",
            "BACF",
            "CSRDCF",
            "DaSiamRPN",
            "ECO",
            "ECO_new",
            "MDNet",
            "SAMF",
            "SiamDW",
            "SiamFC",
            "SiamRPN",
            "Staple",
            "STRCF",
            "TADT",
            "Vital",
        ]
    else:
        trackers = args.experts

    main(trackers, args.dataset_name)
