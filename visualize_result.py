import sys
import os
from pathlib import Path
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from algorithms.aaa_util import calc_overlap

sys.path.append("external/pytracking")

from pytracking.evaluation.environment import env_settings

sns.set()
sns.set_style("whitegrid")


def draw_offline_tracking(dataset, algorithm, result_dir):
    for seq in dataset:
        gt_traj = np.array(seq.ground_truth_rect)
        results_dir = "{}/{}".format(env_settings().results_path, algorithm)
        base_results_path = "{}/{}".format(results_dir, seq.name)
        offline_path = "{}_offline.pkl".format(base_results_path)
        with open(offline_path, "rb") as fp:
            offline_bb = pickle.load(fp)

        results = [gt_traj[0].tolist()]
        for box in offline_bb:
            if box is not None:
                results += list(box)

        save_dir = result_dir / "Offline" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(results)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")

            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            ax.imshow(np.asarray(im), aspect="auto")

            if frame > 1:
                color = "w" if offline_bb[frame - 1] is None else "r"
            else:
                color = "w"

            box = results[frame]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                alpha=1,
            )
            ax.add_patch(rect)
            ax.annotate(
                "Offline",
                xy=(box[0], box[1]),
                xycoords="data",
                xytext=(-50, 10),
                textcoords="offset points",
                size=10,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color),
            )

            box = gt_traj[frame]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
                alpha=1,
            )
            ax.add_patch(rect)
            ax.annotate(
                "Ground Truth",
                xy=(box[0], box[1]),
                xycoords="data",
                xytext=(-50, 10),
                textcoords="offset points",
                size=10,
                color="b",
                arrowprops=dict(arrowstyle="->", color="b"),
            )
            ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()


def draw_result(dataset, algorithm, experts, result_dir):
    colors = sns.color_palette("hls", len(experts) + 1).as_hex()
    trackers = experts + [algorithm]
    for seq in dataset:
        gt_traj = np.array(seq.ground_truth_rect)
        tracker_trajs = []

        for tracker in trackers:
            results_dir = "{}/{}".format(env_settings().results_path, tracker)
            base_results_path = "{}/{}".format(results_dir, seq.name)
            results_path = "{}.txt".format(base_results_path)
            tracker_traj = np.loadtxt(results_path, delimiter="\t")
            tracker_trajs.append(tracker_traj)
        tracker_trajs = np.array(tracker_trajs)

        results_dir = "{}/{}".format(env_settings().results_path, algorithm)
        base_results_path = "{}/{}".format(results_dir, seq.name)
        weights_path = "{}_weight.txt".format(base_results_path)
        tracker_weight = np.loadtxt(weights_path, delimiter="\t")

        offline_path = "{}_offline.pkl".format(base_results_path)
        with open(offline_path, "rb") as fp:
            offline_bb = pickle.load(fp)

        results = [gt_traj[0].tolist()]
        for box in offline_bb:
            if box is not None:
                results += list(box)

        save_dir = result_dir / "Online" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(gt_traj)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")

            fig, axes = plt.subplots(
                nrows=3, ncols=1, gridspec_kw={"height_ratios": [1, 1, 3]}
            )
            fig.add_subplot(111, frameon=False)

            error_ax = axes[0]
            weight_ax = axes[1]
            sample_ax = axes[2]

            # draw error graph
            for i in range(len(trackers)):
                box = tracker_trajs[i, :frame]
                gt = gt_traj[:frame]
                error = 1 - calc_overlap(gt, box)
                error_ax.plot(
                    range(len(error)),
                    error,
                    color=colors[i],
                    label=trackers[i],
                    linewidth=2 if trackers[i].startswith("AAA") else 1,
                )
                error_ax.set(ylabel="Error", xlim=(0, len(gt_traj)), ylim=(-0.05, 1.05))
            error_ax.set_xticks([])
            error_ax.legend(ncol=4, frameon=False, bbox_to_anchor=(1.02, 1.4))

            # draw weight graph
            for i in range(len(tracker_weight)):
                weight = tracker_weight[i]
                weight_ax.plot(
                    range(len(weight)), weight, color=colors[i], label=experts[i]
                )
                weight_ax.set(
                    ylabel="Weight", xlim=(0, len(tracker_weight)), ylim=(-0.05, 1.05)
                )
            weight_ax.set_xticks([])

            # draw anchor line
            for i in range(frame):
                if offline_bb[i - 1] is not None:
                    weight_ax.axvline(x=i, color="gray", linestyle="--", linewidth=0.1)

            # draw frame
            sample_ax.imshow(np.asarray(im), aspect="auto")

            # draw tracking bbox
            for i in range(len(trackers)):
                box = tracker_trajs[i, frame]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    linewidth=2,
                    edgecolor=colors[i],
                    facecolor="none",
                    alpha=1,
                )
                sample_ax.add_patch(rect)
                sample_ax.annotate(
                    "AAA" if trackers[i].startswith("AAA") else trackers[i],
                    xy=(box[0], box[1]),
                    xycoords="data",
                    xytext=(-50, 10),
                    textcoords="offset points",
                    size=10,
                    color=colors[i],
                    arrowprops=dict(arrowstyle="->", color=colors[i]),
                )

            box = gt_traj[frame]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
                alpha=1,
            )
            sample_ax.add_patch(rect)
            sample_ax.annotate(
                "Ground Truth",
                xy=(box[0], box[1]),
                xycoords="data",
                xytext=(-50, 10),
                textcoords="offset points",
                size=10,
                color="b",
                arrowprops=dict(arrowstyle="->", color="b"),
            )
            sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(save_dir / filename)
            plt.close()


def main(experts, algorithm, result_dir):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets = [otb, nfs, uav, tpl, vot, lasot]
    trackers = experts + [algorithm]

    for dataset in datasets:
        draw_offline_tracking(dataset, algorithm, result_dir)
        draw_result(dataset, trackers, result_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algoritm", default="AAA", type=str)
    parser.add_argument("-t", "--trackers", default=list(), nargs="+")
    parser.add_argument("-d", "--dir", default="Expert", type=str)
    args = parser.parse_args()

    result_dir = Path(f"./Result/{args.dir}")
    os.makedirs(result_dir, exist_ok=True)

    main(args.trackers, args.algorithm, result_dir)
