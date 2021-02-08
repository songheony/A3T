import os
from PIL import Image

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtrans
from matplotlib.ticker import MultipleLocator

from algorithms.aaa_util import calc_overlap
from evaluations import ope_benchmark
from visualizes.draw_tables import get_mean_succ, get_mean_ratio, calc_rank

plt.rcParams.update({"font.size": 12})
sns.set()
sns.set_style("whitegrid")
LINE_WIDTH = 2
BOX_WIDTH = 10
ANNO_SIZE = 20


def diffmean(x):
    return x - np.mean(x)


def is_algorithm(tracker_name):
    for algorithm in ["HDT", "MCCT", "Random", "Max", "AAA"]:
        if tracker_name.startswith(algorithm):
            return True
    return False


def draw_pie(
    datasets_name,
    experts_name,
    specific_names,
    success_rets,
    figsize,
    save_dir,
    legend=False,
    file_name=None,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=(len(datasets_name) + 1) // 2,
        figsize=figsize,
        subplot_kw=dict(aspect="equal"),
    )
    fig.add_subplot(111, frameon=False)

    def label(pct):
        if pct > 10:
            return f"{pct:.0f}%"
        else:
            return ""

    for i, dataset_name in enumerate(datasets_name):
        n_row = i % 2
        n_col = i // 2
        ax = axes[n_row, n_col]

        seq_names = sorted(success_rets[dataset_name][experts_name[0]].keys())
        ranks = calc_rank(dataset_name, seq_names, experts_name, success_rets)

        lines, _, _ = ax.pie(
            [np.sum(rank == 1) / len(seq_names) * 100 for rank in ranks.T],
            autopct=label,
            radius=1.2,
        )

        ax.set_title(dataset_name)

    # hide tick and tick label of the big axes
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.grid(False)

    if legend:
        fig.legend(
            lines,
            specific_names,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(-0.1, 0.5),
        )

    if file_name is None:
        file_name = "rank_legend" if legend else "rank"

    plt.subplots_adjust(wspace=-0.2, hspace=0.2)
    plt.savefig(save_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_curves(
    datasets,
    algorithm_name,
    experts_name,
    color_map,
    success_rets,
    precision_rets,
    figsize,
    save_dir,
    file_name=None,
):
    trackers_name = experts_name + [algorithm_name]
    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    for i, dataset_name in enumerate(datasets):
        # draw success plot
        ax = axes[0, i]

        # draw curve of trackers
        thresholds = np.arange(0, 1.05, 0.05)
        for tracker_name in trackers_name:
            value = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            vis_name = (
                tracker_name.split("_")[0]
                if is_algorithm(tracker_name)
                else tracker_name
            )
            line = ax.plot(
                thresholds,
                np.mean(value, axis=0),
                label=vis_name,
                linewidth=LINE_WIDTH * 2 if is_algorithm(tracker_name) else LINE_WIDTH,
                color=color_map[vis_name],
            )[0]

            if i == 0:
                lines.append(line)

        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        if i == 0:
            ax.set_ylabel("Success")
        ax.set_title(dataset_name)

    for i, dataset_name in enumerate(datasets):
        # draw precision plot
        ax = axes[1, i]

        # draw curve of trackers
        thresholds = np.arange(0, 51, 1)
        for tracker_name in trackers_name:
            value = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            vis_name = (
                tracker_name.split("_")[0]
                if is_algorithm(tracker_name)
                else tracker_name
            )
            ax.plot(
                thresholds,
                np.mean(value, axis=0),
                label=vis_name,
                linewidth=LINE_WIDTH * 2 if is_algorithm(tracker_name) else LINE_WIDTH,
                color=color_map[vis_name],
            )

        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        if i == 0:
            ax.set_ylabel("Precision")

    # hide tick and tick label of the big axes
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.grid(False)
    fig.text(0.5125, 0.47, "Threshold for IoU", ha="center", va="center")
    plt.xlabel("Threshold for distance")

    changed_trackers = [tracker_name.split("_")[0] for tracker_name in trackers_name]
    fig.legend(
        lines,
        changed_trackers,
        frameon=False,
        loc="upper center",
        ncol=len(changed_trackers),  # ncol=(len(changed_trackers) + 1) // 2
    )

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    if file_name is None:
        file_name = "curves"
    plt.savefig(save_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()
    plt.close()


def draw_rank(
    datasets,
    group_names,
    group_trackers,
    group_success_rets,
    color_map,
    figsize,
    save_dir,
    legend=False,
    file_name=None,
):
    fig, axes = plt.subplots(
        nrows=len(group_names), ncols=len(datasets), figsize=figsize
    )
    fig.add_subplot(111, frameon=False)

    alphabets = ["a", "b", "c"]

    for g in range(len(group_names)):
        xs = list(range(1, len(group_trackers[g]) + 1))
        for i, dataset_name in enumerate(datasets):
            ax = axes[g, i]

            seq_names = sorted(
                group_success_rets[g][dataset_name][group_trackers[g][0]].keys()
            )
            ranks = calc_rank(
                dataset_name, seq_names, group_trackers[g], group_success_rets[g]
            )

            for tracker_name, rank in zip(group_trackers[g], ranks.T):
                vis_name = (
                    tracker_name.split("/")[0]
                    if is_algorithm(tracker_name)
                    else tracker_name
                )
                ax.plot(
                    xs,
                    [np.sum(rank == x) / len(seq_names) for x in xs],
                    c=color_map[vis_name],
                    label=vis_name,
                    linewidth=LINE_WIDTH * 2
                    if is_algorithm(tracker_name)
                    else LINE_WIDTH,
                )

            if i == len(datasets) // 2 and legend:
                ax.legend(
                    frameon=False,
                    loc="upper center",
                    ncol=len(group_trackers[g]),
                    bbox_to_anchor=(0.0, 1.2),
                )

            # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            # ax.set_xticks([1, len(group_trackers[g]) // 2 + 1, len(group_trackers[g])])
            ax.set_xticks([1, len(group_trackers[g])])
            if g == len(group_names) - 1:
                ax.set_xticklabels(
                    [
                        "1(Best)",
                        # str(len(group_trackers[g]) // 2 + 1),
                        f"{len(group_trackers[g])}(Worst)",
                    ]
                )
            else:
                ax.set_xticklabels([])
            if g == 0:
                ax.set_title(dataset_name, y=1.2)
            if i == 0:
                ax.set_ylabel(f"({alphabets[g]}) {group_names[g]} group")

    # hide tick and tick label of the big axes
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.grid(False)
    plt.xlabel("Rank")
    plt.ylabel("Frequency of rank", labelpad=30)

    plt.subplots_adjust(hspace=0.2)

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = (
        np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
    )
    ymin = (
        np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    )
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0.13, 0.9], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    if file_name is None:
        file_name = "rank_legend" if legend else "rank"
    plt.savefig(save_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_succ_with_thresholds(
    modes,
    thresholds,
    success_rets,
    anchor_frames,
    gt_trajs,
    figsize,
    save_dir,
    file_name=None,
):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    fig.add_subplot(111, frameon=False)

    lines = []

    mean_succ = get_mean_succ(thresholds, modes, success_rets)
    mean_ratio = get_mean_ratio(thresholds, modes, anchor_frames, gt_trajs)

    for i, mode_name in enumerate(modes):
        # draw success plot
        ax = axes[0]

        values = diffmean(mean_succ[:, i])
        # values = mean_succ[:, i]
        line = ax.plot(thresholds, values, label=mode_name, linewidth=LINE_WIDTH)[0]
        lines.append(line)

        max_threshold = np.argmax(values)
        ax.plot(
            thresholds[max_threshold],
            values[max_threshold],
            "o",
            ms=10,
            mec=line.get_color(),
            mfc="none",
            mew=2,
        )

    axes[0].set_ylabel("Diff AUC")

    if anchor_frames is not None:
        for i, mode_name in enumerate(modes):
            # draw highest score
            ax = axes[1]

            ax.plot(
                thresholds,
                mean_ratio[:, i],
                c=lines[i].get_color(),
                label=mode_name,
                linewidth=LINE_WIDTH,
            )
        axes[1].set_ylabel("Anchor ratio")

    # hide tick and tick label of the big axes
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.grid(False)
    plt.xlabel("Threshold")

    fig.legend(lines, modes, frameon=False, loc="upper center", ncol=len(modes))

    plt.subplots_adjust(wspace=0, hspace=0.1)

    if file_name is None:
        file_name = "score_ratio"
    plt.savefig(save_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_result(
    dataset,
    dataset_name,
    algorithm_name,
    experts_name,
    color_map,
    save_dir,
    target_seqs=None,
    show=["legend", "error", "weight", "frame"],
    vis_gt=True,
    vis_best=False,
):
    ope = ope_benchmark.OPEBenchmark(dataset, dataset_name)

    if algorithm_name is not None:
        trackers_name = experts_name + [algorithm_name]
    else:
        trackers_name = experts_name

    for seq in dataset:
        if (target_seqs is not None) and (seq.name not in target_seqs):
            continue

        seq_dir = save_dir / seq.name
        os.makedirs(seq_dir, exist_ok=True)

        gt_traj = np.array(seq.ground_truth_rect)
        tracker_trajs = np.array(
            [
                ope.get_tracker_traj(seq.name, tracker_name)
                for tracker_name in trackers_name
            ]
        )

        if algorithm_name is not None:
            offline_bb, tracker_weight = ope.get_algorithm_data(
                seq.name, algorithm_name
            )

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
                for i in range(len(trackers_name)):
                    box = tracker_trajs[i, :frame]
                    error = 1 - calc_overlap(gt_traj[:frame], box)
                    tracker_name = (
                        trackers_name[i].split("/")[0]
                        if is_algorithm(trackers_name[i])
                        else trackers_name[i]
                    )
                    error_ax.plot(
                        range(len(error)),
                        error,
                        color=color_map[tracker_name],
                        label=tracker_name,
                        linewidth=2 if is_algorithm(trackers_name[i]) else 1,
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
                        range(len(weight)), weight, color=color_map[trackers_name[i]]
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

                if algorithm_name.startswith("AAA"):
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
                    for i in range(len(trackers_name)):
                        tracker_name = trackers_name[i].split("/")[0]
                        box = tracker_trajs[i, frame]
                        rect = patches.Rectangle(
                            (box[0], box[1]),
                            box[2],
                            box[3],
                            linewidth=BOX_WIDTH,
                            edgecolor=color_map[tracker_name],
                            facecolor="none",
                            alpha=1,
                        )
                        sample_ax.add_patch(rect)

                        if algorithm_name is not None:
                            if (frame > 0) and (
                                np.argmax(tracker_weight[frame - 1]) == i
                            ):
                                sample_ax.annotate(
                                    tracker_name,
                                    xy=(box[0], box[1]),
                                    xycoords="data",
                                    weight="bold",
                                    xytext=(-50, 10),
                                    textcoords="offset points",
                                    size=ANNO_SIZE,
                                    color=color_map[tracker_name],
                                    arrowprops=dict(
                                        arrowstyle="->", color=color_map[tracker_name],
                                    ),
                                )
                            elif trackers_name[i] == algorithm_name:
                                sample_ax.annotate(
                                    tracker_name,
                                    xy=(box[0] + box[2], box[1]),
                                    xycoords="data",
                                    weight="bold",
                                    xytext=(10, 10),
                                    textcoords="offset points",
                                    size=ANNO_SIZE,
                                    color=color_map[tracker_name],
                                    arrowprops=dict(
                                        arrowstyle="->", color=color_map[tracker_name],
                                    ),
                                )
                        elif vis_best:
                            scores = calc_overlap(
                                gt_traj[frame], tracker_trajs[:, frame]
                            )
                            if np.argmax(scores) == i:
                                sample_ax.annotate(
                                    tracker_name,
                                    xy=(box[0], box[1]),
                                    xycoords="data",
                                    weight="bold",
                                    xytext=(-50, 10),
                                    textcoords="offset points",
                                    size=ANNO_SIZE,
                                    color=color_map[trackers_name[i]],
                                    arrowprops=dict(
                                        arrowstyle="->",
                                        color=color_map[trackers_name[i]],
                                    ),
                                )

                    if vis_gt:
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
            plt.savefig(seq_dir / filename, bbox_inches="tight")
            plt.close()


def draw_graph(
    dataset,
    dataset_name,
    algorithm_name,
    experts_name,
    color_map,
    save_dir,
    target_seqs=None,
    iserror=True,
    legend=False,
    sframes=[],
):
    ope = ope_benchmark.OPEBenchmark(dataset, dataset_name)
    trackers_name = experts_name + [algorithm_name]

    error_rets = {}
    loss_rets = {}
    for tracker_name in trackers_name:
        error_ret, loss_ret = ope.eval_loss(algorithm_name, tracker_name)
        error_rets[tracker_name] = error_ret
        loss_rets[tracker_name] = loss_ret

    for seq in dataset:
        if (target_seqs is not None) and (seq.name not in target_seqs):
            continue

        seq_dir = save_dir / seq.name
        os.makedirs(save_dir, exist_ok=True)

        offline_bb, tracker_weight = ope.get_algorithm_data(seq.name, algorithm_name)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 1))
        fig.add_subplot(111, frameon=False)

        # draw error graph
        if iserror:
            for i in range(len(trackers_name)):
                error = error_rets[trackers_name[i]][seq.name]
                ax.plot(
                    range(len(error)),
                    error,
                    color=color_map[tracker_name],
                    label=tracker_name,
                    linewidth=LINE_WIDTH * 2
                    if is_algorithm(tracker_name)
                    else LINE_WIDTH,
                    alpha=0.8 if is_algorithm(tracker_name) else 1,
                )
                ax.set(ylabel="Error", xlim=(0, len(error)), ylim=(-0.05, 1.05))
            ax.set_xticks([])
            if legend:
                ax.legend(
                    ncol=len(trackers_name), frameon=False, bbox_to_anchor=(0.2, 1.1)
                )

        # draw weight graph
        else:
            for i in range(tracker_weight.shape[1]):
                weight = tracker_weight[:, i]
                ax.plot(
                    range(len(weight)),
                    weight,
                    color=color_map[trackers_name[i]],
                    linewidth=LINE_WIDTH,  # label=experts[i]
                )
                ax.set(
                    ylabel="Weight",
                    xlim=(0, len(tracker_weight)),
                    ylim=(np.min(tracker_weight) - 0.05, np.max(tracker_weight) + 0.05),
                )
            ax.set_xticks([])

        if algorithm_name.startswith("AAA"):
            # draw anchor line
            for i in range(len(tracker_weight)):
                if offline_bb[i - 1] is not None:
                    ax.axvline(
                        x=i, color="gray", linestyle="--", linewidth=0.1, alpha=0.3
                    )

        for sframe, text in sframes:
            # draw text
            ax.axvline(x=sframe, color="black", linestyle="-", linewidth=1.0)
            if iserror:
                ax.annotate(
                    text,
                    xy=(sframe, 1),
                    xytext=(0, 10),
                    color="black",
                    textcoords="offset points",
                    size=12,
                    ha="center",
                    va="center",
                )

        filename = "error" if iserror else "weight"
        if legend:
            filename += "_legend"
        filename += ".pdf"

        # hide tick and tick label of the big axes
        plt.axis("off")
        plt.grid(False)
        plt.savefig(seq_dir / filename, bbox_inches="tight")
        plt.close()
