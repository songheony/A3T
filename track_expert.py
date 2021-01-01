from track_dataset import run
from select_options import select_expert


def main(tracker_name, dataset_name):
    tracker = select_expert(tracker_name)

    run(tracker, dataset_name, experts=None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert", default="RPT", type=str, help="expert")
    parser.add_argument("-d", "--dataset", default="OTB2015", type=str, help="dataset")
    args = parser.parse_args()

    main(args.expert, args.dataset)
