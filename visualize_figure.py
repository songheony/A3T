import os
from pathlib import Path
import pickle
from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from visualize_result import draw_graph, draw_result
from visualize_eval import ALGORITHMS, name2color


def main(experts, baselines, algorithm, eval_dir, result_dir):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets = [otb, tpl, uav, nfs, lasot, vot]
    datasets_name = ["OTB", "TPL", "UAV", "NFS", "LaSOT", "VOT"]

    eval_save = eval_dir / "eval.pkl"
    successes, precisions, anchor_frames, anchor_successes, anchor_precisions, offline_successes, offline_precisions = pickle.loads(
        eval_save.read_bytes()
    )

    eval_hard = ["Basketball", "Bird1", "tpl_SuperMario_ce"]
    good_exam = ["Girl2", "tpl_Yo_yos_ce1"]
    mcct_bad_seq = ["Bird1", "Tiger1", "Soccer"]

    for dataset, dataset_name in zip(datasets, datasets_name):
        dataset_dir = result_dir / dataset_name
        # draw_result(dataset, None, experts, name2color(experts), dataset_dir, eval_hard, show=["frame"], gt=False)

        for algorithm_name in baselines + [algorithm]:
            algorithm_colors = name2color(experts + [algorithm])
            if algorithm.startswith("AAA"):
                draw_graph(
                    dataset,
                    algorithm,
                    experts,
                    algorithm_colors,
                    dataset_dir,
                    [good_exam[0]],
                    iserror=True,
                    legend=True,
                )
                draw_graph(
                    dataset,
                    algorithm,
                    experts,
                    algorithm_colors,
                    dataset_dir,
                    [good_exam[1]],
                    iserror=True,
                    legend=False,
                )
                draw_graph(
                    dataset,
                    algorithm,
                    experts,
                    algorithm_colors,
                    dataset_dir,
                    good_exam,
                    iserror=False,
                    legend=False,
                )
                draw_result(
                    dataset,
                    algorithm,
                    experts,
                    algorithm_colors,
                    dataset_dir,
                    good_exam,
                    show=["frame"],
                )
            elif algorithm.startswith("MCCT"):
                draw_result(
                    dataset,
                    algorithm,
                    experts,
                    algorithm_colors,
                    dataset_dir,
                    mcct_bad_seq,
                    show=["frame"],
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-b", "--baselines", default=list(), nargs="+")
    parser.add_argument("-d", "--dir", default="Expert", type=str)
    args = parser.parse_args()

    result_dir = Path(f"./visualize_results/{args.dir}")
    os.makedirs(result_dir, exist_ok=True)

    eval_dir = Path(f"./evaluation_results/{args.dir}")

    main(args.experts, args.baselines, args.algorithm, eval_dir, result_dir)
