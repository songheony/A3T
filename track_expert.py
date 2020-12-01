import random
import numpy as np
import torch
from track_dataset import run
from options import select_expert

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)


def main(tracker_name, dataset_name):
    tracker = select_expert(tracker_name)

    run(tracker, dataset_name, experts=None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert", default="SiamFC++", type=str, help="expert")
    parser.add_argument("-d", "--dataset", default="OTB", type=str, help="dataset")
    args = parser.parse_args()

    main(args.expert, args.dataset)
