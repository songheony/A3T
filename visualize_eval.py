import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.transforms as mtrans
from sympy import preview

from datasets.lasotdataset import LaSOTDataset
from datasets.nfsdataset import NFSDataset
from datasets.otbdataset import OTBDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.votdataset import VOTDataset
from evaluations.offline_benchmark import OfflineBenchmark
from evaluations.ope_benchmark import OPEBenchmark

plt.rcParams.update({"font.size": 12})
sns.set()
sns.set_style("whitegrid")
EXPERTS = [
    "ATOM",
    "DaSiamRPN",
    "GradNet",
    "MemTrack",
    "SiamDW",
    "SiamFC",
    "SiamMCF",
    "SiamRPN",
    "SiamRPN++",
    "SPM",
    "Staple",
    "THOR",
]
ALGORITHMS = ["HDT", "MCCT", "Random", "Max", "AAA"]
COLORS = sns.color_palette("hls", len(EXPERTS) + 2).as_hex()[::-1][1:]
LINE_WIDTH = 2


def isalgorithm(tracker):
    for algo in ALGORITHMS:
        if tracker.startswith(algo):
            return True

    return False


def name2color(trackers):
    color = []
    expert_idx = 0
    for tracker in trackers:
        if isalgorithm(tracker):
            color += [COLORS[-1]]
        else:
            if "_" in tracker:
                color += [COLORS[expert_idx]]
                expert_idx += 1
            else:
                idx = EXPERTS.index(tracker)
                color += [COLORS[idx]]
    return color


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def calc_rank(rets, trackers, dataset_name, seq_names):
    ranks = []
    for seq_name in seq_names:
        value = np.mean(
            [rets[dataset_name][tracker_name][seq_name] for tracker_name in trackers],
            axis=1,
        )
        temp = value.argsort()
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(value))
        rank = len(trackers) - rank
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks


def draw_score_anchor(datasets, algorithm_name, success_rets, anchor_frames, figsize, eval_dir):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    for i, dataset_name in enumerate(datasets):
        seq_names = sorted(success_rets[dataset_name][algorithm_name].keys())

        # draw curve of trackers
        ratio = [
            sum(anchor_frames[dataset_name][seq_name]) / len(anchor_frames[dataset_name][seq_name]) if not np.any(np.isnan(anchor_frames[dataset_name][seq_name])) else 0 for seq_name in seq_names
        ]
        value = [
            np.mean(success_rets[dataset_name][algorithm_name][seq_name]) if not any(np.isnan(success_rets[dataset_name][algorithm_name][seq_name])) else 0
            for seq_name in seq_names
        ]
        ax.scatter(
            ratio,
            value,
            label=dataset_name,
        )
    ax.set_ylabel("AUC")
    ax.set_xlabel("Anchor ratio")

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
    fig.legend(
        frameon=False,
        loc="upper center",
        ncol=len(datasets)
    )
    plt.savefig(eval_dir / "auc_ratio.pdf", bbox_inches="tight")
    plt.close()


def draw_curves(datasets, trackers, success_rets, precision_rets, figsize, eval_dir, file_name=None):
    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    for i, dataset_name in enumerate(datasets):
        # draw success plot
        ax = axes[0, i]

        # draw curve of trackers
        thresholds = np.arange(0, 1.05, 0.05)
        for tracker_name in trackers:
            value = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            line = ax.plot(
                thresholds,
                np.mean(value, axis=0),
                label=tracker_name.split("_")[0]
                if isalgorithm(tracker_name)
                else tracker_name,
                linewidth=LINE_WIDTH * 2 if isalgorithm(tracker_name) else LINE_WIDTH,
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
        for tracker_name in trackers:
            value = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            ax.plot(
                thresholds,
                np.mean(value, axis=0),
                label=tracker_name.split("_")[0]
                if isalgorithm(tracker_name)
                else tracker_name,
                linewidth=LINE_WIDTH * 2 if isalgorithm(tracker_name) else LINE_WIDTH,
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
    fig.text(0.5125, 0.47, 'Threshold for IoU', ha='center', va='center')
    plt.xlabel("Threshold for distance")

    changed_trackers = [
        tracker_name if "AAA" not in tracker_name else "AAA"
        for tracker_name in trackers
    ]
    fig.legend(
        lines,
        changed_trackers,
        frameon=False,
        loc="upper center",
        ncol=len(trackers),  # ncol=(len(trackers) + 1) // 2
    )

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    if file_name is None:
        file_name = "curves"
    plt.savefig(eval_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()
    plt.close()


def draw_rank(
    datasets, trackers, success_rets, figsize, eval_dir, legend=False, file_name=None
):
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    xs = list(range(1, len(trackers) + 1))

    for i, dataset_name in enumerate(datasets):
        # draw success plot
        ax = axes[i]

        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(success_rets, trackers, dataset_name, seq_names)

        for tracker_name, rank in zip(trackers, ranks.T):
            line = ax.plot(
                xs,
                [np.sum(rank == x) / len(seq_names) for x in xs],
                # c=color,
                label=tracker_name.split("_")[0]
                if isalgorithm(tracker_name)
                else tracker_name,
                linewidth=LINE_WIDTH * 2 if isalgorithm(tracker_name) else LINE_WIDTH,
            )[0]
            if i == 0:
                lines.append(line)

        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xticks([1, len(trackers) // 2 + 1, len(trackers)])
        ax.set_xticklabels(["1(Best)", str(len(trackers) // 2 + 1), f"{len(trackers)}(Worst)"])
        if i == 0:
            ax.set_ylabel("Frequency of rank")
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
    plt.xlabel("Rank")

    if legend:
        changed_trackers = [
            tracker_name.split("_")[0] if isalgorithm(tracker_name) else tracker_name
            for tracker_name in trackers
        ]
        fig.legend(
            lines,
            changed_trackers,
            frameon=False,
            loc="upper center",
            ncol=len(trackers),
            # bbox_to_anchor=[0.4, 1.4],
        )

    plt.subplots_adjust(wspace=0.2, top=0.8)

    if file_name is None:
        file_name = "rank_legend" if legend else "rank"
    plt.savefig(eval_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_rank_all(
    datasets, group_names, group_trackers, group_success_rets, all_trackers, figsize, eval_dir, legend=False, file_name=None
):
    fig, axes = plt.subplots(nrows=len(group_names), ncols=len(datasets), figsize=figsize)
    fig.add_subplot(111, frameon=False)

    xs = list(range(1, len(group_trackers[0]) + 1))

    alphabets = ["a", "b", "c"]

    for g in range(len(group_names)):
        for i, dataset_name in enumerate(datasets):
            ax = axes[g, i]

            seq_names = sorted(group_success_rets[g][dataset_name][group_trackers[g][0]].keys())
            ranks = calc_rank(group_success_rets[g], group_trackers[g], dataset_name, seq_names)

            for tracker_name, rank in zip(group_trackers[g], ranks.T):
                ax.plot(
                    xs,
                    [np.sum(rank == x) / len(seq_names) for x in xs],
                    c=name2color([tracker_name])[0],
                    label=tracker_name.split("_")[0]
                    if isalgorithm(tracker_name)
                    else tracker_name,
                    linewidth=LINE_WIDTH * 2 if isalgorithm(tracker_name) else LINE_WIDTH,
                )

            if i == len(datasets) // 2 and legend:
                ax.legend(
                    frameon=False,
                    loc="upper center",
                    ncol=len(group_trackers[g]),
                    bbox_to_anchor=(0.0, 1.2)
                )

            # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.set_xticks([1, len(group_trackers[g]) // 2 + 1, len(group_trackers[g])])
            if g == len(group_names) - 1:
                ax.set_xticklabels(["1(Best)", str(len(group_trackers[g]) // 2 + 1), f"{len(group_trackers[g])}(Worst)"])
            else:
                ax.set_xticklabels([])
            if g == 0:
                ax.set_title(dataset_name, y=1.2)
            # if i == len(datasets) - 1:
            #     ax.yaxis.set_label_position("right")
            #     ax.set_ylabel(f"({alphabets[g]}) In {group_names[g]} group")
            if i == 0:
                # ax.yaxis.set_label_position("right")
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
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0.13, 0.9], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    if file_name is None:
        file_name = "rank_legend" if legend else "rank"
    plt.savefig(eval_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_rank_both(
    datasets, trackers, success_rets, figsize, eval_dir, legend=False, file_name=None
):
    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    xs = list(range(1, len(trackers) + 1))

    for i, dataset_name in enumerate(datasets):
        # draw success plot
        ax = axes[0, i]

        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(success_rets, trackers, dataset_name, seq_names)

        for tracker_name, rank in zip(trackers, ranks.T):
            line = ax.plot(
                xs,
                [np.sum(rank == x) / len(seq_names) for x in xs],
                # c=color,
                label=tracker_name.split("_")[0]
                if isalgorithm(tracker_name)
                else tracker_name,
                linewidth=LINE_WIDTH * 2 if isalgorithm(tracker_name) else LINE_WIDTH,
            )[0]
            if i == 0:
                lines.append(line)

        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xticks([1, len(trackers) // 2 + 1, len(trackers)])
        ax.set_xticklabels([])
        if i == 0:
            ax.set_ylabel("Frequency of rank")
        ax.set_title(dataset_name)

    for i, dataset_name in enumerate(datasets):
        # draw precision plot
        ax = axes[1, i]

        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(success_rets, trackers, dataset_name, seq_names)

        for tracker_name, rank in zip(trackers, ranks.T):
            ax.plot(
                xs,
                np.cumsum([np.sum(rank == x) for x in xs]) / len(seq_names),
                # c=color,
                label=tracker_name.split("_")[0]
                if isalgorithm(tracker_name)
                else tracker_name,
                linewidth=LINE_WIDTH * 2 if isalgorithm(tracker_name) else LINE_WIDTH,
            )

        # ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xticks([1, len(trackers) // 2 + 1, len(trackers)])
        ax.set_xticklabels(["Best", str(len(trackers) // 2 + 1), "Worst"])
        if i == 0:
            ax.set_ylabel("Cumulrative frequency")

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

    if legend:
        changed_trackers = [
            tracker_name.split("_")[0] if isalgorithm(tracker_name) else tracker_name
            for tracker_name in trackers
        ]
        fig.legend(
            lines,
            changed_trackers,
            frameon=False,
            loc="upper center",
            ncol=len(trackers),  # ncol=(len(trackers) + 1) // 2
        )

    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    if file_name is None:
        file_name = "rank_both_legend" if legend else "rank_both"
    plt.savefig(eval_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_pie(
    datasets,
    trackers,
    trackers_name,
    success_rets,
    figsize,
    save_dir,
    legend=False,
    file_name=None,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(datasets) // 2,
        figsize=figsize,
        subplot_kw=dict(aspect="equal"),
    )
    fig.add_subplot(111, frameon=False)

    def label(pct):
        if pct > 10:
            return f"{pct:.0f}%"
        else:
            return ""

    for i, dataset_name in enumerate(datasets):
        n_row = i % 2
        n_col = i % (len(datasets) // 2)
        ax = axes[n_row, n_col]

        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(success_rets, trackers, dataset_name, seq_names)

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
            trackers_name,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5)
            # ncol=len(trackers) // 3
        )

    if file_name is None:
        file_name = "rank_legend" if legend else "rank"

    plt.subplots_adjust(wspace=-0.2, hspace=0.2)
    plt.savefig(save_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_succ_with_thresholds(
    modes, thresholds, success_rets, anchor_frames, figsize, eval_dir, file_name=None
):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    fig.add_subplot(111, frameon=False)

    lines = []

    mean_succ = np.zeros((len(thresholds), len(modes)))
    mean_ratio = np.zeros((len(thresholds), len(modes)))
    for i, tracker_name in enumerate(thresholds):
        for j, dataset_name in enumerate(modes):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)

            if anchor_frames is not None:
                ratio = [
                    sum(v) / len(v)
                    for v in anchor_frames[dataset_name][tracker_name].values()
                    if not np.any(np.isnan(v))
                ]
                mean_ratio[i, j] = np.mean(ratio)
            else:
                mean_ratio[i, j] = 1

    for i, mode_name in enumerate(modes):
        # draw success plot
        ax = axes[0]

        line = ax.plot(
            thresholds, minmax(mean_succ[:, i]), label=mode_name, linewidth=LINE_WIDTH
        )[0]
        lines.append(line)

        ax.plot(
            thresholds[np.argmax(mean_succ[:, i])],
            1,
            "o",
            ms=10,
            mec=line.get_color(),
            mfc="none",
            mew=2,
        )

    axes[0].set_ylabel("Normalized AUC")

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
    plt.savefig(eval_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def draw_prec_with_thresholds(
    modes, thresholds, precision_rets, anchor_frames, figsize, eval_dir, file_name=None
):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    fig.add_subplot(111, frameon=False)

    lines = []

    mean_prec = np.zeros((len(thresholds), len(modes)))
    mean_ratio = np.zeros((len(thresholds), len(modes)))
    for i, tracker_name in enumerate(thresholds):
        for j, dataset_name in enumerate(modes):
            prec = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_prec[i, j] = np.mean(prec, axis=0)[20]

            if anchor_frames is not None:
                ratio = [
                    sum(v) / len(v)
                    for v in anchor_frames[dataset_name][tracker_name].values()
                    if not np.any(np.isnan(v))
                ]
                mean_ratio[i, j] = np.mean(ratio)
            else:
                mean_ratio[i, j] = 1

    for i, mode_name in enumerate(modes):
        # draw success plot
        ax = axes[0]

        line = ax.plot(
            thresholds, minmax(mean_prec[:, i]), label=mode_name, linewidth=LINE_WIDTH
        )[0]
        lines.append(line)

        ax.plot(
            thresholds[np.argmax(mean_prec[:, i])],
            1,
            "o",
            ms=10,
            mec=line.get_color(),
            mfc="none",
            mew=2,
        )

    axes[0].set_ylabel("Normalized DP")

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
    plt.savefig(eval_dir / f"{file_name}.pdf", bbox_inches="tight")
    plt.close()


def find_rank(datasets, algorithms, experts, success_rets, eval_dir):
    text = ""
    for i, dataset_name in enumerate(datasets):
        text += f"{dataset_name}\n"
        text += "-" * 10 + "\n"
        seq_names = sorted(success_rets[dataset_name][experts[0]].keys())
        for algorithm in algorithms:
            rank = calc_rank(
                success_rets, experts + [algorithm], dataset_name, seq_names
            )[:, -1]
            first = [seq for seq, cond in zip(seq_names, rank == 1) if cond]
            last = [
                seq for seq, cond in zip(seq_names, rank == len(experts) + 1) if cond
            ]
            text += f"{algorithm.split('_')[0]}: best-{first} / worst-{last}\n"
        text += "\n"

    txt_file = eval_dir / f"Ranking.txt"
    txt_file.write_text(text)


def make_score_table(
    datasets,
    trackers,
    nexperts,
    success_rets,
    precision_rets,
    eval_dir,
    filename=None,
    isvot=False,
):
    mean_succ = np.zeros((len(trackers), len(datasets)))
    mean_prec = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)
            prec = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_prec[i, j] = np.mean(prec, axis=0)[20]

    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c"
    for i in range(len(datasets) * 2 - 1 if isvot else len(datasets) * 2):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Tracker}"

    for i in range(len(datasets) - 1):
        columns += f" & \\multicolumn{{2}}{{c|}}{{{datasets[i]}}}"
    if isvot:
        columns += f" & {datasets[-1]}"
    else:
        columns += f" & \\multicolumn{{2}}{{c|}}{{{datasets[-1]}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(datasets) - 1):
        small_columns += " & AUC & DP"
    if isvot:
        small_columns += " & AO"
    else:
        small_columns += " & AUC & DP"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers)):
        if i == nexperts:
            latex += "\\hdashline\n"

        if (i >= nexperts) and ("_" in trackers[i]):
            line = trackers[i][: trackers[i].index("_")]
        else:
            line = trackers[i].replace("_", "\\_")

        if isvot:
            for j in range(len(datasets) - 1):
                for value in [mean_succ, mean_prec]:
                    sorted_idx = np.argsort(value[:, j])
                    if i == sorted_idx[-1]:
                        line += f' & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}'
                    elif i == sorted_idx[-2]:
                        line += f' & {{\\color{{blue}} \\textit{{{value[i, j]:0.2f}}}}}'
                    else:
                        line += f" & {value[i, j]:0.2f}"

            vot_idx = len(datasets) - 1
            sorted_idx = np.argsort(mean_succ[:, vot_idx])
            if i == sorted_idx[-1]:
                line += (
                    f' & {{\\color{{red}} \\textbf{{{mean_succ[i, vot_idx]:0.2f}}}}}'
                )
            elif i == sorted_idx[-2]:
                line += (
                    f' & {{\\color{{blue}} \\textit{{{mean_succ[i, vot_idx]:0.2f}}}}}'
                )
            else:
                line += f" & {mean_succ[i, vot_idx]:0.2f}"
        else:
            for j in range(len(datasets)):
                for value in [mean_succ, mean_prec]:
                    sorted_idx = np.argsort(value[:, j])
                    if i == sorted_idx[-1]:
                        line += f' & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}'
                    elif i == sorted_idx[-2]:
                        line += f' & {{\\color{{blue}} \\textit{{{value[i, j]:0.2f}}}}}'
                    else:
                        line += f" & {value[i, j]:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = eval_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln", "threeparttable"),
    )


def make_regret_table(
    datasets, trackers, nexperts, regret_offlines, eval_dir, filename=None
):
    mean_offline = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            regret_offline = [
                v
                for v in regret_offlines[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_offline[i, j] = np.mean(regret_offline)

    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c"
    for i in range(len(datasets)):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "Tracker"

    for i in range(len(datasets)):
        columns += f" & {datasets[i]}"
    latex += f"{columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers)):
        if i == nexperts:
            latex += "\\hdashline\n"

        if (i >= nexperts) and ("_" in trackers[i]):
            line = trackers[i][: trackers[i].index("_")]
        else:
            line = trackers[i].replace("_", "\\_")

        for j in range(len(datasets)):
            sorted_idx = np.argsort(mean_offline[:, j])[::-1]
            if i == sorted_idx[-1]:
                line += f' & {{\\color{{red}} \\textbf{{{mean_offline[i, j]:0.2f}}}}}'
            elif i == sorted_idx[-2]:
                line += f' & {{\\color{{blue}} \\textit{{{mean_offline[i, j]:0.2f}}}}}'
            else:
                line += f" & {mean_offline[i, j]:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = eval_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / f"{filename}.png",
        packages=("xcolor", "arydshln", "threeparttable"),
    )


def make_score_regret_table(
    datasets,
    trackers,
    nexperts,
    success_rets,
    precision_rets,
    regret_offlines,
    eval_dir,
    filename=None,
    isvot=False,
):
    mean_succ = np.zeros((len(trackers), len(datasets)))
    mean_prec = np.zeros((len(trackers), len(datasets)))
    mean_offline = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)
            prec = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_prec[i, j] = np.mean(prec, axis=0)[20]
            regret_offline = [
                v
                for v in regret_offlines[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_offline[i, j] = np.mean(regret_offline)

    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c"
    for i in range(len(datasets) * 3 - 1 if isvot else len(datasets) * 3):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Tracker}"

    for i in range(len(datasets) - 1):
        columns += f" & \\multicolumn{{3}}{{c}}{{{datasets[i]}}}"
    if isvot:
        columns += f" & \\multicolumn{{2}}{{c}}{{{datasets[-1]}}}"
    else:
        columns += f" & \\multicolumn{{3}}{{c}}{{{datasets[-1]}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(datasets) - 1):
        small_columns += " & AUC & DP & R"
    if isvot:
        small_columns += " & AO & R"
    else:
        small_columns += " & AUC & DP & R"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers)):
        if i == nexperts:
            latex += "\\hdashline\n"

        if (i >= nexperts) and ("_" in trackers[i]):
            line = trackers[i][: trackers[i].index("_")]
        else:
            line = trackers[i].replace("_", "\\_")

        if isvot:
            for j in range(len(datasets) - 1):
                for value in [mean_succ, mean_prec]:
                    sorted_idx = np.argsort(value[:, j])
                    if i == sorted_idx[-1]:
                        line += f' & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}'
                    elif i == sorted_idx[-2]:
                        line += f' & {{\\color{{blue}} \\textit{{{value[i, j]:0.2f}}}}}'
                    else:
                        line += f" & {value[i, j]:0.2f}"
                sorted_idx = np.argsort(mean_offline[:, j])[::-1]
                if i == sorted_idx[-1]:
                    line += (
                        f' & {{\\color{{red}} \\textbf{{{mean_offline[i, j]:0.2f}}}}}'
                    )
                elif i == sorted_idx[-2]:
                    line += (
                        f' & {{\\color{{blue}} \\textit{{{mean_offline[i, j]:0.2f}}}}}'
                    )
                else:
                    line += f" & {mean_offline[i, j]:0.2f}"

            vot_idx = len(datasets) - 1
            sorted_idx = np.argsort(mean_succ[:, vot_idx])
            if i == sorted_idx[-1]:
                line += (
                    f' & {{\\color{{red}} \\textbf{{{mean_succ[i, vot_idx]:0.2f}}}}}'
                )
            elif i == sorted_idx[-2]:
                line += (
                    f' & {{\\color{{blue}} \\textit{{{mean_succ[i, vot_idx]:0.2f}}}}}'
                )
            else:
                line += f" & {mean_succ[i, vot_idx]:0.2f}"

            sorted_idx = np.argsort(mean_offline[:, vot_idx])[::-1]
            if i == sorted_idx[-1]:
                line += (
                    f' & {{\\color{{red}} \\textbf{{{mean_offline[i, vot_idx]:0.2f}}}}}'
                )
            elif i == sorted_idx[-2]:
                line += f' & {{\\color{{blue}} \\textit{{{mean_offline[i, vot_idx]:0.2f}}}}}'
            else:
                line += f" & {mean_offline[i, vot_idx]:0.2f}"
        else:
            for j in range(len(datasets)):
                for value in [mean_succ, mean_prec]:
                    sorted_idx = np.argsort(value[:, j])
                    if i == sorted_idx[-1]:
                        line += f' & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}'
                    elif i == sorted_idx[-2]:
                        line += f' & {{\\color{{blue}} \\textit{{{value[i, j]:0.2f}}}}}'
                    else:
                        line += f" & {value[i, j]:0.2f}"

                sorted_idx = np.argsort(mean_offline[:, j])[::-1]
                if i == sorted_idx[-1]:
                    line += (
                        f' & {{\\color{{red}} \\textbf{{{mean_offline[i, j]:0.2f}}}}}'
                    )
                elif i == sorted_idx[-2]:
                    line += (
                        f' & {{\\color{{blue}} \\textit{{{mean_offline[i, j]:0.2f}}}}}'
                    )
                else:
                    line += f" & {mean_offline[i, j]:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = eval_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln", "threeparttable"),
    )


def make_ratio_table(
    datasets, trackers, success_rets, anchor_frames, eval_dir, filename=None
):
    mean_succ = np.zeros((len(trackers), len(datasets)))
    mean_ratio = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)

            if anchor_frames is not None:
                ratio = [
                    sum(v) / len(v)
                    for v in anchor_frames[dataset_name][tracker_name].values()
                    if not np.any(np.isnan(v))
                ]
                mean_ratio[i, j] = np.mean(ratio)
            else:
                mean_ratio[i, j] = 1

    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c"
    for i in range(len(datasets) * 2):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Threshold}"

    for i in range(len(datasets)):
        columns += f" & \\multicolumn{{2}}{{c}}{{{datasets[i]}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(datasets)):
        small_columns += " & AUC & Ratio"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers)):
        line = trackers[i]

        for j in range(len(datasets)):
            for value in [mean_succ, mean_ratio]:
                sorted_idx = np.argsort(value[:, j])
                if i == sorted_idx[-1]:
                    line += f' & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}'
                elif i == sorted_idx[-2]:
                    line += f' & {{\\color{{blue}} \\textit{{{value[i, j]:0.2f}}}}}'
                else:
                    line += f" & {value[i, j]:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = eval_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln", "threeparttable"),
    )


def draw_scores(datasets, trackers, success_rets, eval_dir, filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))

    ind = np.arange(len(datasets)) * 2
    width = 0.25

    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
    plt.grid(False)

    mean_succ = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)

    lines = []
    for idx, tracker_name in enumerate(trackers):
        value = mean_succ[idx]
        label = "Ours" if "AAA" in tracker_name else tracker_name
        line = ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            value,
            width,
            label=label,
        )
        lines.append(line)

    ax.set_ylabel("AUC")
    ax.set_xticks(ind)
    ax.set_xticklabels(datasets)
    ax.set_ylim([0.3, 0.75])

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
    plt.xlabel("Dataset")

    fig.legend(frameon=False, loc="upper center", ncol=len(trackers))

    if filename is None:
        filename = "score"

    plt.savefig(eval_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close()


def main(experts, baselines, algorithm, eval_dir):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets = [otb, tpl, uav, nfs, lasot, vot]
    datasets_name = ["OTB2015", "TColor128", "UAV123", "NFS", "LaSOT", "VOT2018"]
    if algorithm is not None:
        eval_trackers = experts + baselines + [algorithm]
        # viz_trackers = experts + [algorithm]
    else:
        eval_trackers = experts
        # viz_trackers = experts

    eval_save = eval_dir / "eval.pkl"
    if eval_save.exists():
        successes, precisions, anchor_frames, anchor_successes, anchor_precisions, offline_successes, offline_precisions, regret_gts, regret_offlines = pickle.loads(
            eval_save.read_bytes()
        )
    else:
        successes = {}
        precisions = {}
        anchor_frames = {}
        anchor_successes = {}
        anchor_precisions = {}
        offline_successes = {}
        offline_precisions = {}
        regret_gts = {}
        regret_offlines = {}

        for dataset, name in zip(datasets, datasets_name):
            ope = OPEBenchmark(dataset)
            offline = OfflineBenchmark(dataset)

            success = ope.eval_success(eval_trackers)
            precision = ope.eval_precision(eval_trackers)
            successes[name] = success
            precisions[name] = precision

            if algorithm is not None:
                anchor_frame, anchor_success, anchor_precision = offline.eval_anchor_frame(
                    algorithm, experts
                )
                offline_success, offline_precision = offline.eval_offline_tracker(
                    algorithm, experts
                )
                regret_gt, regret_offline = offline.eval_regret(
                    algorithm, eval_trackers, experts
                )

                anchor_frames[name] = anchor_frame
                anchor_successes[name] = anchor_success
                anchor_precisions[name] = anchor_precision
                offline_successes[name] = offline_success
                offline_precisions[name] = offline_precision
                regret_gts[name] = regret_gt
                regret_offlines[name] = regret_offline

        eval_save.write_bytes(
            pickle.dumps(
                (
                    successes,
                    precisions,
                    anchor_frames,
                    anchor_successes,
                    anchor_precisions,
                    offline_successes,
                    offline_precisions,
                    regret_gts,
                    regret_offlines,
                )
            )
        )

    find_rank(datasets_name, baselines + [algorithm], experts, successes, eval_dir)

    # colors = name2color(eval_trackers)
    # sns.set_palette(colors)

    # make_score_table(
    #     datasets_name,
    #     eval_trackers,
    #     len(experts),
    #     successes,
    #     precisions,
    #     eval_dir,
    #     "Score",
    #     isvot=True,
    # )
    # figsize = (20, 5)
    # draw_curves(datasets_name, viz_trackers, successes, precisions, figsize, eval_dir)
    # draw_rank_both(datasets_name, viz_trackers, successes, figsize, eval_dir)
    # draw_rank_both(
    #     datasets_name, viz_trackers, successes, figsize, eval_dir, legend=True
    # )

    # if algorithm is not None:
    #     find_rank(datasets_name, baselines + [algorithm], experts, successes, eval_dir)

    #     make_score_table(
    #         datasets_name,
    #         viz_trackers,
    #         len(experts),
    #         anchor_successes,
    #         anchor_precisions,
    #         eval_dir,
    #         "Anchor",
    #         isvot=True,
    #     )

    #     make_score_table(
    #         datasets_name,
    #         viz_trackers,
    #         len(experts),
    #         offline_successes,
    #         offline_precisions,
    #         eval_dir,
    #         "Offline",
    #         isvot=True,
    #     )

    #     make_regret_table(
    #         datasets_name,
    #         viz_trackers,
    #         len(experts),
    #         regret_offlines,
    #         eval_dir,
    #         "Regret",
    #     )
    # else:
    #     figsize = (10, 5)
    #     draw_pie(datasets_name, viz_trackers, successes, figsize, eval_dir, legend=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default=None, type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-b", "--baselines", default=list(), nargs="+")
    parser.add_argument("-d", "--dir", default="Expert", type=str)
    args = parser.parse_args()

    eval_dir = Path(f"./evaluation_results/{args.dir}")
    os.makedirs(eval_dir, exist_ok=True)

    main(args.experts, args.baselines, args.algorithm, eval_dir)
