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
                results += box

        save_dir = result_dir / "Offline" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(results)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")
            box = results[frame]

            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            ax.imshow(np.asarray(im), aspect="auto")

            if frame > 1:
                color = "w" if offline_bb[frame - 1] is None else "r"
            else:
                color = "w"

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
            ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.xlabel("Anchor ratio")
            plt.ylabel("AUC")

            plt.savefig(os.path.join(save_dir, filename))
            plt.close()


def draw_result(dataset, trackers, result_dir):
    colors = sns.color_palette("hls", len(trackers) + 1).as_hex()
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

        save_dir = result_dir / "Online" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(gt_traj)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")

            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            ax.imshow(np.asarray(im), aspect="auto")

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
                ax.add_patch(rect)
                ax.annotate(
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
            plt.xlabel("Anchor ratio")
            plt.ylabel("AUC")

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
    experts = [
        "ATOM",
        "DaSiamRPN",
        "ECO",
        "SiamDW",
        "SiamMCF",
        "SiamRPN++",
        "SPM",
        "THOR",
    ]

    result_dir = Path("./Tracking")

    main(experts, "AAA")
