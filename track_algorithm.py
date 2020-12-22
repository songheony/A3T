import random
import numpy as np
import torch
from track_dataset import run
from select_options import select_algorithms

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)


def main(algorithm_name, experts, dataset_name, **kwargs):
    algorithm = select_algorithms(algorithm_name, experts, **kwargs)

    run(algorithm, dataset_name, experts=experts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-d", "--dataset", default="OTB", type=str)
    parser.add_argument("-n", "--mode", default="High", type=str)
    parser.add_argument("-r", "--threshold", default=0.0, type=float)
    args = parser.parse_args()

    main(
        args.algorithm,
        args.experts,
        args.dataset,
        mode=args.mode,
        threshold=args.threshold,
    )
