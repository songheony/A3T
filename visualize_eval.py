import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from sympy import preview

from datasets.lasotdataset import LaSOTDataset
from datasets.nfsdataset import NFSDataset
from datasets.otbdataset import OTBDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.votdataset import VOTDataset
from evaluations.offline_benchmark import OfflineBenchmark
from evaluations.ope_benchmark import OPEBenchmark

sns.set()
sns.set_style("whitegrid")


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


def draw_scores(datasets, trackers, success_rets, precision_rets, figsize, eval_dir):
    ind = np.arange(len(datasets)) * 2
    width = 0.15

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []

    # draw Success score
    ax = axes[0]
    for idx, tracker_name in enumerate(trackers):
        values = []
        for dataset_name in datasets:
            value = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            values.append(np.mean(value))
        line = ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            values,
            width,
            label=tracker_name if "AAA" not in tracker_name else "AAA",
        )
        lines.append(line)
    ax.set_ylabel("AUC of Success")
    ax.set_xticks(ind)

    # draw Precision score
    ax = axes[1]
    for idx, tracker_name in enumerate(trackers):
        values = []
        for dataset_name in datasets:
            value = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            values.append(np.mean(value, axis=0)[20])
        ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            values,
            width,
            label=tracker_name if "AAA" not in tracker_name else "AAA",
        )
    ax.set_ylabel("Median of Precision")
    ax.set_xticks(ind)
    ax.set_xticklabels(datasets)

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

    changed_trackers = [
        tracker_name if "AAA" not in tracker_name else "AAA"
        for tracker_name in trackers
    ]
    fig.legend(
        lines, changed_trackers, loc="upper center", ncol=(len(trackers) + 1) // 2
    )
    plt.savefig(eval_dir / f"scores.pdf", bbox_inches="tight")
    plt.close()


def draw_curves(datasets, trackers, success_rets, precision_rets, figsize, eval_dir):
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
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=2,
            )[0]

            if i == 0:
                lines.append(line)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if i == 0:
            ax.set_ylabel("Success")

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
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=2,
            )

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if i == 0:
            ax.set_ylabel("Precision")
        ax.set_xlabel(dataset_name)

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
    plt.xlabel("\nThreshold")

    changed_trackers = [
        tracker_name if "AAA" not in tracker_name else "AAA"
        for tracker_name in trackers
    ]
    fig.legend(
        lines, changed_trackers, loc="upper center", ncol=(len(trackers) + 1) // 2
    )

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(eval_dir / "curves.pdf", bbox_inches="tight")
    plt.close()


def draw_rank_both(datasets, trackers, success_rets, figsize, eval_dir):
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
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=8 if "AAA" in tracker_name else 4,
            )[0]
            if i == 0:
                lines.append(line)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks([1, len(trackers) // 2 + 1, len(trackers)])
        ax.set_xticklabels([])
        if i == 0:
            ax.set_ylabel("Frequency of rank")

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
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=8 if "AAA" in tracker_name else 4,
            )

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks([1, len(trackers) // 2 + 1, len(trackers)])
        ax.set_xticklabels(["Best", str(len(trackers) // 2 + 1), "Worst"], fontsize=15)
        if i == 0:
            ax.set_ylabel("Cumulrative frequency")
        ax.set_xlabel(dataset_name)

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
    plt.xlabel("\nRank")

    # changed_trackers = [
    #     tracker_name if "AAA" not in tracker_name else "AAA"
    #     for tracker_name in trackers
    # ]
    # fig.legend(
    #     lines, changed_trackers, loc="upper center", ncol=(len(trackers) + 1) // 2
    # )

    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.savefig(eval_dir / "rank_both.pdf", bbox_inches="tight")
    plt.close()


def draw_anchor_ratio_score(
    datasets, algorithm, success_rets, anchor_frames, figsize, eval_dir
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    legend_names = []

    for i, dataset_name in enumerate(datasets):
        seq_names = sorted(success_rets[dataset_name][algorithm].keys())
        succs = []
        ratio = []
        for seq in seq_names:
            succs.append(np.mean(success_rets[dataset_name][algorithm][seq]))
            valid_idx = anchor_frames[dataset_name][seq]
            ratio.append(sum(valid_idx) / len(valid_idx))
        line = ax.scatter(
            ratio, succs, label=f"{dataset_name}[{np.mean(ratio):.2f}]"
        )  # c=colors[i]
        lines.append(line)
        legend_names.append(f"{dataset_name}[{np.mean(ratio):.2f}]")
    ax.set_ylabel("AUC")

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
    plt.xlabel("Anchor ratio")

    fig.legend(lines, legend_names, loc="upper center", ncol=len(datasets))

    plt.savefig(eval_dir / f"anchor_ratio_score.pdf", bbox_inches="tight")
    plt.close()


def draw_anchor_ratio_rank(
    datasets, trackers, success_rets, anchor_frames, figsize, eval_dir
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    legend_names = []

    for i, dataset_name in enumerate(datasets):
        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        rank = calc_rank(success_rets, trackers, dataset_name, seq_names)[:, -1]
        ratio = []
        for seq in seq_names:
            valid_idx = anchor_frames[dataset_name][seq]
            ratio.append(sum(valid_idx) / len(valid_idx))
        line = ax.scatter(
            ratio, rank, label=f"{dataset_name}[{np.mean(ratio):.2f}]"
        )  # c=colors[i]
        lines.append(line)
        legend_names.append(f"{dataset_name}[{np.mean(ratio):.2f}]")
    ax.set_ylabel("Rank")

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
    plt.xlabel("Anchor ratio")

    fig.legend(lines, legend_names, loc="upper center", ncol=len(datasets))

    plt.savefig(eval_dir / f"anchor_ratio_rank.pdf", bbox_inches="tight")
    plt.close()


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
                if not any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)
            prec = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            mean_prec[i, j] = np.mean(prec, axis=0)[20]

    latex = "\\begin{table*}\n"
    latex += "\\begin{center}\n"
    latex += "\\begin{small}\n"

    header = "c"
    for i in range(len(datasets) * 2 - 1 if isvot else len(datasets) * 2):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Threshold}"

    for i in range(len(datasets) - 1):
        columns += f" & \\multicolumn{{2}}{{c}}{{{datasets[i]}}}"
    if isvot:
        columns += f" & {datasets[-1]}"
    else:
        columns += f" & \\multicolumn{{2}}{{c}}{{{datasets[-1]}}}"
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
                        line += f' & {{\\color{{blue}} \\textbf{{{value[i, j]:0.2f}}}}}'
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
                    f' & {{\\color{{blue}} \\textbf{{{mean_succ[i, vot_idx]:0.2f}}}}}'
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
                        line += f' & {{\\color{{blue}} \\textbf{{{value[i, j]:0.2f}}}}}'
                    else:
                        line += f" & {value[i, j]:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{small}\n"
    latex += "\\end{center}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = eval_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln"),
    )


def make_ratio_table(
    datasets, trackers, nexperts, success_rets, anchor_ratios, eval_dir, filename=None
):
    mean_succ = np.zeros((len(trackers), len(datasets)))
    mean_ratio = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)
            ratio = [
                sum(v) / len(v)
                for v in anchor_ratios[dataset_name].values()
                if not any(np.isnan(v))
            ]
            mean_ratio[i, j] = np.mean(ratio)

    latex = "\\begin{table*}\n"
    latex += "\\begin{center}\n"
    latex += "\\begin{small}\n"

    header = "c"
    for i in range(len(datasets) * 2):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Tracker}"

    for i in range(len(datasets)):
        columns += f" & \\multicolumn{{2}}{{c}}{{{datasets[i]}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(datasets)):
        small_columns += " & AUC & Ratio"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers)):
        if i == nexperts:
            latex += "\\hdashline\n"

        if (i >= nexperts) and ("_" in trackers[i]):
            line = trackers[i][: trackers[i].index("_")]
        else:
            line = trackers[i].replace("_", "\\_")

        for j in range(len(datasets)):
            for value in [mean_succ, mean_ratio]:
                sorted_idx = np.argsort(value[:, j])
                if i == sorted_idx[-1]:
                    line += f' & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}'
                elif i == sorted_idx[-2]:
                    line += f' & {{\\color{{blue}} \\textbf{{{value[i, j]:0.2f}}}}}'
                else:
                    line += f" & {value[i, j]:0.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{small}\n"
    latex += "\\end{center}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = eval_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln"),
    )


def main(experts, baselines, algorithm, eval_dir):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets_name = ["OTB", "NFS", "UAV", "TPL", "LaSOT", "VOT"]
    datasets = [otb, nfs, uav, tpl, vot, lasot]
    eval_trackers = experts + baselines + [algorithm]
    viz_trackers = experts + [algorithm]

    eval_save = eval_dir / "eval.pkl"
    if eval_save.exists():
        successes, precisions, anchor_frames, anchor_successes, anchor_precisions, offline_successes, offline_precisions = pickle.loads(
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

        for dataset, name in zip(datasets, datasets_name):
            ope = OPEBenchmark(dataset)
            offline = OfflineBenchmark(dataset)

            success = ope.eval_success(eval_trackers)
            precision = ope.eval_precision(eval_trackers)
            anchor_frame, anchor_success, anchor_precision = offline.eval_anchor_frame(
                algorithm, experts
            )
            offline_success, offline_precision = offline.eval_offline_tracker(
                algorithm, experts
            )

            successes[name] = success
            precisions[name] = precision
            anchor_frames[name] = anchor_frame
            anchor_successes[name] = anchor_success
            anchor_precisions[name] = anchor_precision
            offline_successes[name] = offline_success
            offline_precisions[name] = offline_precision

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
                )
            )
        )

    make_score_table(
        datasets_name,
        eval_trackers,
        len(experts),
        successes,
        precisions,
        eval_dir,
        "Score",
        isvot=True,
    )

    make_score_table(
        datasets_name,
        experts + ["Anchor"],
        len(experts),
        anchor_successes,
        anchor_precisions,
        eval_dir,
        "Anchor",
        isvot=True,
    )

    make_score_table(
        datasets_name,
        experts + ["Offline tracker"],
        len(experts),
        offline_successes,
        offline_precisions,
        eval_dir,
        "Offline",
        isvot=True,
    )

    colors = sns.color_palette("hls", len(datasets) + 1).as_hex()
    sns.set_palette(colors)

    figsize = (10, 3)
    draw_anchor_ratio_score(
        datasets_name, algorithm, successes, anchor_frames, figsize, eval_dir
    )

    draw_anchor_ratio_rank(
        datasets_name, viz_trackers, successes, anchor_frames, figsize, eval_dir
    )

    colors = sns.color_palette("hls", len(viz_trackers) + 1).as_hex()[::-1][1:]
    sns.set_palette(colors)

    figsize = (20, 6)
    draw_curves(datasets_name, viz_trackers, successes, precisions, figsize, eval_dir)
    draw_rank_both(datasets_name, viz_trackers, successes, figsize, eval_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-b", "--baselines", default=list(), nargs="+")
    parser.add_argument("-d", "--dir", default="Expert", type=str)
    args = parser.parse_args()

    eval_dir = Path(f"./evaluation_results/{args.dir}")
    os.makedirs(eval_dir, exist_ok=True)

    main(args.experts, args.baselines, args.algorithm, eval_dir)
