from track_dataset import run
from select_options import select_algorithms


def main(algorithm_name, experts, dataset_name, **kwargs):
    algorithm = select_algorithms(algorithm_name, experts, **kwargs)

    run(algorithm, dataset_name, experts=experts, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument(
        "-e", "--experts", default=["DaSiamRPN", "SiamDW", "SiamRPN", "SPM"], nargs="+"
    )
    parser.add_argument("-d", "--dataset", default="OTB2015", type=str)
    parser.add_argument("-m", "--mode", default="SuperFast", type=str)
    parser.add_argument("-t", "--threshold", default=0.8, type=float)
    args = parser.parse_args()

    main(
        args.algorithm,
        args.experts,
        args.dataset,
        mode=args.mode,
        threshold=args.threshold,
    )
