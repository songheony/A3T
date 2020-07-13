import sys
import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from algorithms.aaa_util import calc_overlap

sys.path.append("external/pytracking")

from pytracking.evaluation.environment import env_settings

sns.set()
sns.set_style("whitegrid")

plt.rcParams.update({"font.size": 30})

BOX_WIDTH = 10
ANNO_SIZE = 15
LINE_WIDTH = 3


def draw_groudtruth(dataset, result_dir):
    for seq in dataset:
        gt_traj = np.array(seq.ground_truth_rect)

        save_dir = result_dir / "GroundTruth" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(gt_traj)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")

            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)

            ax.imshow(np.asarray(im), aspect="auto")

            box = gt_traj[frame]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=BOX_WIDTH,
                edgecolor="r",
                facecolor="none",
                alpha=1,
            )
            ax.add_patch(rect)
            ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()


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
                linewidth=BOX_WIDTH,
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
                size=ANNO_SIZE,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color),
            )

            box = gt_traj[frame]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=BOX_WIDTH,
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
                size=ANNO_SIZE,
                color="b",
                arrowprops=dict(arrowstyle="->", color="b"),
            )
            ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()


def draw_result(
    dataset,
    algorithm,
    experts,
    colors,
    result_dir,
    target_seqs=None,
    show=["legend", "error", "weight", "frame"],
    gt=True,
    best=False,
):
    trackers = experts if algorithm is None else experts + [algorithm]
    for seq in dataset:
        if (target_seqs is not None) and (seq.name not in target_seqs):
            continue
        gt_traj = np.array(seq.ground_truth_rect)
        tracker_trajs = []

        for tracker in trackers:
            results_dir = "{}/{}".format(env_settings().results_path, tracker)
            base_results_path = "{}/{}".format(results_dir, seq.name)
            results_path = "{}.txt".format(base_results_path)
            tracker_traj = np.loadtxt(results_path, delimiter="\t")
            tracker_trajs.append(tracker_traj)
        tracker_trajs = np.array(tracker_trajs)

        if algorithm is None:
            save_dir = result_dir / "Experts" / seq.name
        else:
            results_dir = "{}/{}".format(env_settings().results_path, algorithm)
            base_results_path = "{}/{}".format(results_dir, seq.name)
            weights_path = "{}_weight.txt".format(base_results_path)
            tracker_weight = np.loadtxt(weights_path, delimiter="\t")

            offline_path = "{}_offline.pkl".format(base_results_path)
            with open(offline_path, "rb") as fp:
                offline_bb = pickle.load(fp)

            save_dir = result_dir / f"{algorithm.split('_')[0]}" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(gt_traj)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")

            cond = [drawing in show for drawing in ["error", "weight", "frame"]]
            ratios = [1 if i != 2 else 3 for i in range(len(cond)) if cond[i]]

            fig, axes = plt.subplots(
                nrows=sum(cond), ncols=1, gridspec_kw={"height_ratios": ratios}
            )
            fig.add_subplot(111, frameon=False)

            i = 0
            if cond[0]:
                error_ax = axes[i] if len(ratios) > 1 else axes
                i += 1
            if cond[1]:
                weight_ax = axes[i] if len(ratios) > 1 else axes
                i += 1
            if cond[2]:
                sample_ax = axes[i] if len(ratios) > 1 else axes

            # draw error graph
            if cond[0]:
                for i in range(len(trackers)):
                    box = tracker_trajs[i, :frame]
                    error = 1 - calc_overlap(gt_traj[:frame], box)
                    error_ax.plot(
                        range(len(error)),
                        error,
                        color=colors[i],
                        label=trackers[i].split("_")[0]
                        if i == len(trackers) - 1
                        else trackers[i],
                        linewidth=2 if i == len(trackers) - 1 else 1,
                    )
                    error_ax.set(
                        ylabel="Error", xlim=(0, len(gt_traj)), ylim=(-0.05, 1.05)
                    )
                error_ax.set_xticks([])
                if "legend" in show:
                    error_ax.legend(ncol=4, frameon=False, bbox_to_anchor=(1.02, 1.8))

            # draw weight graph
            if cond[1]:
                for i in range(tracker_weight.shape[1]):
                    weight = tracker_weight[:frame, i]
                    weight_ax.plot(
                        range(len(weight)), weight, color=colors[i]  # label=experts[i]
                    )
                    weight_ax.set(
                        ylabel="Weight",
                        xlim=(0, len(tracker_weight)),
                        ylim=(
                            np.min(tracker_weight) - 0.05,
                            np.max(tracker_weight) + 0.05,
                        ),
                    )
                weight_ax.set_xticks([])

                if algorithm.startswith("AAA"):
                    # draw anchor line
                    for i in range(frame):
                        if offline_bb[i - 1] is not None:
                            weight_ax.axvline(
                                x=i, color="gray", linestyle="--", linewidth=0.1
                            )

            # draw frame
            if cond[2]:
                sample_ax.imshow(np.asarray(im), aspect="auto")

                if frame == 0:
                    box = gt_traj[frame]
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2],
                        box[3],
                        linewidth=BOX_WIDTH,
                        edgecolor="black",
                        facecolor="none",
                        alpha=1,
                    )
                    sample_ax.add_patch(rect)
                else:
                    # draw tracking bbox
                    for i in range(len(trackers)):
                        box = tracker_trajs[i, frame]
                        rect = patches.Rectangle(
                            (box[0], box[1]),
                            box[2],
                            box[3],
                            linewidth=BOX_WIDTH,
                            edgecolor=colors[i],
                            facecolor="none",
                            alpha=1,
                        )
                        sample_ax.add_patch(rect)

                        if algorithm is not None:
                            if (frame > 0) and (
                                np.argmax(tracker_weight[frame - 1]) == i
                            ):
                                sample_ax.annotate(
                                    trackers[i],
                                    xy=(box[0], box[1]),
                                    xycoords="data",
                                    weight="bold",
                                    xytext=(-50, 10),
                                    textcoords="offset points",
                                    size=ANNO_SIZE,
                                    color=colors[i],
                                    arrowprops=dict(arrowstyle="->", color=colors[i]),
                                )
                            elif trackers[i] == algorithm:
                                sample_ax.annotate(
                                    trackers[i].split("_")[0],
                                    xy=(box[0] + box[2], box[1]),
                                    xycoords="data",
                                    weight="bold",
                                    xytext=(10, 10),
                                    textcoords="offset points",
                                    size=ANNO_SIZE,
                                    color=colors[i],
                                    arrowprops=dict(arrowstyle="->", color=colors[i]),
                                )
                        elif best:
                            scores = calc_overlap(
                                gt_traj[frame], tracker_trajs[:, frame]
                            )
                            if np.argmax(scores) == i:
                                sample_ax.annotate(
                                    trackers[i],
                                    xy=(box[0], box[1]),
                                    xycoords="data",
                                    weight="bold",
                                    xytext=(-50, 10),
                                    textcoords="offset points",
                                    size=ANNO_SIZE,
                                    color=colors[i],
                                    arrowprops=dict(arrowstyle="->", color=colors[i]),
                                )

                    if gt:
                        box = gt_traj[frame]
                        rect = patches.Rectangle(
                            (box[0], box[1]),
                            box[2],
                            box[3],
                            linewidth=BOX_WIDTH,
                            edgecolor="black",
                            facecolor="none",
                            alpha=1,
                        )
                        sample_ax.add_patch(rect)
                        sample_ax.annotate(
                            "Ground Truth",
                            xy=(box[0], box[1] + box[3]),
                            xycoords="data",
                            weight="bold",
                            xytext=(-50, -20),
                            textcoords="offset points",
                            size=ANNO_SIZE,
                            color="black",
                            arrowprops=dict(arrowstyle="->", color="black"),
                        )
                sample_ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.subplots_adjust(wspace=0, hspace=0.1 if len(ratios) > 1 else 0)
            plt.savefig(save_dir / filename, bbox_inches="tight")
            plt.close()


def draw_graph(
    dataset,
    algorithm,
    experts,
    colors,
    result_dir,
    target_seqs=None,
    iserror=False,
    legend=False,
):
    trackers = experts + [algorithm]
    for seq in dataset:
        if (target_seqs is not None) and (seq.name not in target_seqs):
            continue
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

        save_dir = result_dir / f"{algorithm.split('_')[0]}" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 1))
        fig.add_subplot(111, frameon=False)

        # draw error graph
        if iserror:
            for i in range(len(trackers)):
                box = tracker_trajs[i, :]
                error = 1 - calc_overlap(gt_traj, box)
                ax.plot(
                    range(len(error)),
                    error,
                    color=colors[i],
                    label=trackers[i].split("_")[0]
                    if trackers[i].startswith("AAA")
                    else trackers[i],
                    linewidth=LINE_WIDTH * 2
                    if trackers[i].startswith("AAA")
                    else LINE_WIDTH,
                    alpha=0.8 if trackers[i].startswith("AAA") else 1,
                )
                ax.set(ylabel="Error", xlim=(0, len(gt_traj)), ylim=(-0.05, 1.05))
            ax.set_xticks([])
            if legend:
                ax.legend(ncol=len(trackers), frameon=False, bbox_to_anchor=(0.2, 1.0))

        # draw weight graph
        else:
            for i in range(tracker_weight.shape[1]):
                weight = tracker_weight[:, i]
                ax.plot(
                    range(len(weight)),
                    weight,
                    color=colors[i],
                    linewidth=LINE_WIDTH,  # label=experts[i]
                )
                ax.set(
                    ylabel="Weight",
                    xlim=(0, len(tracker_weight)),
                    ylim=(np.min(tracker_weight) - 0.05, np.max(tracker_weight) + 0.05),
                )
            ax.set_xticks([])

        if algorithm.startswith("AAA"):
            # draw anchor line
            for i in range(len(gt_traj)):
                if offline_bb[i - 1] is not None:
                    ax.axvline(
                        x=i, color="gray", linestyle="--", linewidth=0.1, alpha=0.3
                    )

        filename = "error" if iserror else "weight"
        if legend:
            filename += "_legend"
        filename += ".pdf"

        # hide tick and tick label of the big axes
        plt.axis("off")
        plt.grid(False)
        plt.savefig(save_dir / filename, bbox_inches="tight")
        plt.close()


def draw_algorithms(dataset, algorithms, colors, result_dir):
    for seq in dataset:
        gt_traj = np.array(seq.ground_truth_rect)
        tracker_trajs = []

        for tracker in algorithms:
            results_dir = "{}/{}".format(env_settings().results_path, tracker)
            base_results_path = "{}/{}".format(results_dir, seq.name)
            results_path = "{}.txt".format(base_results_path)
            tracker_traj = np.loadtxt(results_path, delimiter="\t")
            tracker_trajs.append(tracker_traj)
        tracker_trajs = np.array(tracker_trajs)

        save_dir = result_dir / "Algorithm" / seq.name
        os.makedirs(save_dir, exist_ok=True)

        for frame in range(len(gt_traj)):
            filename = os.path.basename(seq.frames[frame])
            im = Image.open(seq.frames[frame]).convert("RGB")

            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.add_subplot(111, frameon=False)
            # draw frame
            ax.imshow(np.asarray(im), aspect="auto")

            # draw tracking bbox
            for i in range(len(algorithms)):
                box = tracker_trajs[i, frame]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    linewidth=BOX_WIDTH,
                    edgecolor=colors[i],
                    facecolor="none",
                    alpha=1,
                )
                ax.add_patch(rect)
                ax.annotate(
                    algorithms[i].split("_")[0],
                    xy=(box[0], box[1]),
                    xycoords="data",
                    xytext=(-50, 10),
                    textcoords="offset points",
                    size=ANNO_SIZE,
                    color=colors[i],
                    arrowprops=dict(arrowstyle="->", color=colors[i]),
                )

            box = gt_traj[frame]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=BOX_WIDTH,
                edgecolor="black",
                facecolor="none",
                alpha=1,
            )
            ax.add_patch(rect)
            ax.annotate(
                "Ground Truth",
                xy=(box[0], box[1]),
                xycoords="data",
                xytext=(-50, -20),
                textcoords="offset points",
                size=ANNO_SIZE,
                color="black",
                arrowprops=dict(arrowstyle="->", color="black"),
            )
            ax.axis("off")

            # hide tick and tick label of the big axes
            plt.axis("off")
            plt.grid(False)
            plt.savefig(save_dir / filename)
            plt.close()
