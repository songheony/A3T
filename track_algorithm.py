from track_dataset import run
from select_options import select_algorithms


def main(algorithm_name, experts, dataset_name, **kwargs):
    algorithm = select_algorithms(algorithm_name, experts, **kwargs)

    run(algorithm, dataset_name, experts=experts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-d", "--dataset", default="OTB", type=str)
    parser.add_argument("-m", "--mode", default="High", type=str)
    parser.add_argument("-t", "--threshold", default=0.0, type=float)
    parser.add_argument("-oa", "--offline_a", default=1, type=int)
    parser.add_argument("-ob", "--offline_b", default=1, type=int)
    parser.add_argument("-oc", "--offline_c", default=1, type=int)
    args = parser.parse_args()

    main(
        args.algorithm,
        args.experts,
        args.dataset,
        mode=args.mode,
        threshold=args.threshold,
        offline_a=args.offline_a,
        offline_b=args.offline_b,
        offline_c=args.offline_c,
    )
