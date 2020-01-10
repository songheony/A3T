import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from sympy import preview

from datasets.lasotdataset import LaSOTDataset
from datasets.nfsdataset import NFSDataset
from datasets.otbdataset import OTBDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.votdataset import VOTDataset
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
            value = [v for k, v in success_rets[dataset_name][tracker_name].items()]
            line = ax.plot(
                thresholds,
                np.mean(value, axis=0),
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=2,
            )[0]

            if i == 0:
                lines.append(line)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i == 0:
            ax.set_ylabel("Success")

    for i, dataset_name in enumerate(datasets):
        # draw precision plot
        ax = axes[1, i]

        # draw curve of trackers
        thresholds = np.arange(0, 51, 1)
        for tracker_name in trackers:
            value = [v for k, v in precision_rets[dataset_name][tracker_name].items()]
            ax.plot(
                thresholds,
                np.mean(value, axis=0),
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=2,
            )

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
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


def draw_scores(
    datasets, trackers, success_rets, precision_rets, figsize, eval_dir, norm=None
):
    ind = np.arange(len(datasets)) * 2
    width = 0.15

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []

    # draw Success score
    ax = axes[0]
    succs = {}
    for tracker_name in trackers:
        values = []
        for dataset_name in datasets:
            value = [v for k, v in success_rets[dataset_name][tracker_name].items()]
            values.append(np.mean(value))
        succs[tracker_name] = values
    maxi = np.max(np.array(list(succs.values())), axis=0)
    mini = np.min(np.array(list(succs.values())), axis=0)
    mean = np.mean(np.array(list(succs.values())), axis=0)
    std = np.std(np.array(list(succs.values())), axis=0)
    for idx, tracker_name in enumerate(trackers):
        if norm == "minmax":
            value = (np.array(succs[tracker_name]) - mini) / (maxi - mini)
        elif norm == "std":
            value = (np.array(succs[tracker_name]) - mean) / std
        else:
            value = succs[tracker_name]
        line = ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            value,
            width,
            label=tracker_name if "AAA" not in tracker_name else "AAA",
        )
        lines.append(line)
    ax.set_ylabel("AUC of Success")
    ax.set_xticks(ind)

    # draw Precision score
    ax = axes[1]
    precs = {}
    for tracker_name in trackers:
        values = []
        for dataset_name in datasets:
            value = [v for k, v in precision_rets[dataset_name][tracker_name].items()]
            values.append(np.mean(value, axis=0)[20])
        precs[tracker_name] = values
    maxi = np.max(np.array(list(precs.values())), axis=0)
    mini = np.min(np.array(list(precs.values())), axis=0)
    mean = np.mean(np.array(list(precs.values())), axis=0)
    std = np.std(np.array(list(precs.values())), axis=0)
    for idx, tracker_name in enumerate(trackers):
        if norm == "minmax":
            value = (np.array(precs[tracker_name]) - mini) / (maxi - mini)
        elif norm == "std":
            value = (np.array(precs[tracker_name]) - mean) / std
        else:
            value = precs[tracker_name]
        ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            value,
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

    if norm is None:
        plt.savefig(eval_dir / "scores.pdf", bbox_inches="tight")
    else:
        plt.savefig(eval_dir / f"scores_{norm}.pdf", bbox_inches="tight")
    plt.close()


def draw_dists(datasets, trackers, success_rets, precision_rets, figsize, eval_dir):
    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), sharex=True, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    for i, dataset_name in enumerate(datasets):
        ax = axes[0, i]
        # draw Success rank
        seq_succ = []
        for tracker_name in trackers:
            value = [v for k, v in success_rets[dataset_name][tracker_name].items()]
            seq_succ.append(np.mean(value, axis=1))
        seq_succ = np.array(seq_succ)
        relative = np.amax(seq_succ, axis=0, keepdims=True) - seq_succ
        for idx, tracker_name in enumerate(trackers):
            sns.distplot(
                relative[idx],
                ax=ax,
                hist=False,
                kde=True,
                label=(tracker_name if "AAA" not in tracker_name else "AAA") if i == 0 else None,
            )

        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if i == 0:
            ax.get_legend().remove()
            ax.set_ylabel("Frequency(Success)")

    for i, dataset_name in enumerate(datasets):
        ax = axes[1, i]
        # draw Success rank
        seq_prec = []
        for tracker_name in trackers:
            value = [v for k, v in precision_rets[dataset_name][tracker_name].items()]
            seq_prec.append(np.array(value)[:, 20])
        seq_prec = np.array(seq_prec)
        relative = np.amax(seq_prec, axis=0, keepdims=True) - seq_prec
        for idx, tracker_name in enumerate(trackers):
            sns.distplot(relative[idx], ax=ax, hist=False, kde=True)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if i == 0:
            ax.set_ylabel("Frequency(Precision)")
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
    plt.xlabel("\nDifference from the Best")

    fig.legend(loc="upper center", ncol=(len(trackers) + 1) // 2)

    plt.savefig(eval_dir / "dists.pdf", bbox_inches="tight")
    plt.close()


def draw_score_hists(
    datasets, trackers, success_rets, precision_rets, figsize, eval_dir
):

    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), sharex=True, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = [None] * len(trackers)
    bins = np.arange(0, 1 + 0.05, 0.05)

    # draw Success relative
    for i, dataset_name in enumerate(datasets):
        ax = axes[0, i]
        seq_succ = []
        for tracker_name in trackers:
            value = [v for k, v in success_rets[dataset_name][tracker_name].items()]
            seq_succ.append(np.mean(value, axis=1))
        seq_succ = np.array(seq_succ)
        relative = np.amax(seq_succ, axis=0, keepdims=True) - seq_succ

        for idx, tracker_name in enumerate(trackers):
            hist, _ = np.histogram(relative[idx], bins=bins)
            line = ax.plot(
                bins[:-1] + 0.025,
                hist,
                # c=colors[idx],
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                marker="." if idx == len(trackers) - 1 else None,
                linewidth=3 if idx == len(trackers) - 1 else 1,
            )[0]
            if i == 0:
                lines[idx] = line

        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if i == 0:
            ax.set_ylabel("Frequency(Success)")

    # draw Success relative
    for i, dataset_name in enumerate(datasets):
        ax = axes[1, i]
        seq_prec = []
        for tracker_name in trackers:
            value = [v for k, v in precision_rets[dataset_name][tracker_name].items()]
            seq_prec.append(np.array(value)[:, 20])
        seq_prec = np.array(seq_prec)
        relative = np.amax(seq_prec, axis=0, keepdims=True) - seq_prec

        for idx, tracker_name in enumerate(trackers):
            hist, _ = np.histogram(relative[idx], bins=bins)
            ax.plot(
                bins[:-1] + 0.025,
                hist,
                # c=colors[idx],
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                marker="." if idx == len(trackers) - 1 else None,
                linewidth=3 if idx == len(trackers) - 1 else 1,
            )

        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if i == 0:
            ax.set_ylabel("Frequency(Precision)")
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
    plt.xlabel("\nDifference from the Best")

    changed_trackers = [
        tracker_name if "AAA" not in tracker_name else "AAA"
        for tracker_name in trackers
    ]
    fig.legend(
        lines, changed_trackers, loc="upper center", ncol=(len(trackers) + 1) // 2
    )

    plt.savefig(eval_dir / "score_hists.pdf", bbox_inches="tight")
    plt.close()


def draw_rank_hists(datasets, trackers, success_rets, figsize, eval_dir):
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), sharex=True, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    xs = list(range(1, len(trackers) + 1))

    # draw Success rank
    for i, dataset_name in enumerate(datasets):
        ax = axes[i]
        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(success_rets, trackers, dataset_name, seq_names)

        for tracker_name, rank in zip(trackers, ranks.T):
            line = ax.plot(
                xs,
                [np.sum(rank == x) for x in xs],
                # c=color,
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=8 if "AAA" in tracker_name else 4,
            )[0]
            if i == 0:
                lines.append(line)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if i == 0:
            ax.set_ylabel("# of image sequences")
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

    changed_trackers = [
        tracker_name if "AAA" not in tracker_name else "AAA"
        for tracker_name in trackers
    ]
    fig.legend(
        lines, changed_trackers, loc="upper center", ncol=(len(trackers) + 1) // 2
    )

    plt.savefig(eval_dir / "rank_hists.pdf", bbox_inches="tight")
    plt.close()


def draw_rank_sum(datasets, trackers, success_rets, figsize, eval_dir):
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), sharex=True, figsize=figsize)
    fig.add_subplot(111, frameon=False)

    lines = []
    xs = list(range(1, len(trackers) + 1))

    # draw Success rank
    for i, dataset_name in enumerate(datasets):
        ax = axes[i]
        seq_names = sorted(success_rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(success_rets, trackers, dataset_name, seq_names)

        for tracker_name, rank in zip(trackers, ranks.T):
            line = ax.plot(
                xs,
                [np.sum(rank <= x) for x in xs],
                # c=color,
                label=tracker_name if "AAA" not in tracker_name else "AAA",
                linewidth=8 if "AAA" in tracker_name else 4,
            )[0]
            if i == 0:
                lines.append(line)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        if i == 0:
            ax.set_ylabel("# of image sequences")
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

    changed_trackers = [
        tracker_name if "AAA" not in tracker_name else "AAA"
        for tracker_name in trackers
    ]
    fig.legend(
        lines, changed_trackers, loc="upper center", ncol=(len(trackers) + 1) // 2
    )

    plt.savefig(eval_dir / "rank_sum.pdf", bbox_inches="tight")
    plt.close()


def draw_ratio(
    datasets, algorithm, success_rets, precision_rets, anchor_frames, figsize, eval_dir
):
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=True, figsize=figsize
    )
    fig.add_subplot(111, frameon=False)

    lines = []

    ax = axes[0]
    for i, dataset in enumerate(datasets):
        succs = []
        ratio = []
        for seq in success_rets[dataset][algorithm].keys():
            succs.append(np.mean(success_rets[dataset][algorithm][seq]))
            anchor_frame = anchor_frames[dataset][algorithm][seq]
            ratio.append(sum(anchor_frame) / len(anchor_frame))
        line = ax.scatter(ratio, succs, label=dataset)  # c=colors[i]
        lines.append(line)
    ax.set_ylabel("AUC of Success")

    ax = axes[1]
    for i, dataset in enumerate(datasets):
        precs = []
        ratio = []
        for seq in success_rets[dataset][algorithm].keys():
            precs.append(precision_rets[dataset][algorithm][seq][20])
            anchor_frame = anchor_frames[dataset][algorithm][seq]
            ratio.append(sum(anchor_frame) / len(anchor_frame))
        ax.scatter(ratio, precs, label=dataset)  # c=colors[i]
    ax.set_ylabel("Median of Precision")

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

    fig.legend(lines, datasets, loc="upper center", ncol=(len(datasets) + 1) // 2)

    plt.savefig(eval_dir / f"ratios.pdf", bbox_inches="tight")
    plt.close()


def draw_ratio_rank(
    datasets, trackers, success_rets, precision_rets, anchor_frames, figsize, eval_dir
):
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=figsize
    )
    fig.add_subplot(111, frameon=False)

    lines = []

    ax = axes[0]
    for i, dataset in enumerate(datasets):
        seq_names = sorted(success_rets[dataset][trackers[-1]].keys())
        succs = []
        ratio = []
        for seq in seq_names:
            succs.append(np.mean(success_rets[dataset][trackers[-1]][seq]))
            anchor_frame = anchor_frames[dataset][trackers[-1]][seq]
            ratio.append(sum(anchor_frame) / len(anchor_frame))
        line = ax.scatter(ratio, succs, label=dataset)  # c=colors[i]
        lines.append(line)
    ax.set_ylabel("AUC of Success")

    ax = axes[1]
    for i, dataset in enumerate(datasets):
        seq_names = sorted(success_rets[dataset][trackers[-1]].keys())
        ranks = calc_rank(success_rets, trackers, dataset, seq_names)
        ratio = []
        for seq in seq_names:
            anchor_frame = anchor_frames[dataset][trackers[-1]][seq]
            ratio.append(sum(anchor_frame) / len(anchor_frame))
        ax.scatter(ratio, ranks[:, -1], label=dataset)  # c=colors[i]
    ax.set_ylabel("Rank of AUC")

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

    fig.legend(lines, datasets, loc="upper center", ncol=(len(datasets) + 1) // 2)

    plt.savefig(eval_dir / f"ratios_rank.pdf", bbox_inches="tight")
    plt.close()


def make_table(datasets, trackers, nexperts, success_rets, precision_rets, eval_dir):
    mean_succ = np.zeros((len(trackers), len(datasets)))
    mean_prec = np.zeros((len(trackers), len(datasets)))
    for i, tracker_name in enumerate(trackers):
        for j, dataset_name in enumerate(datasets):
            succ = [v for k, v in success_rets[dataset_name][tracker_name].items()]
            mean_succ[i, j] = np.mean(succ)
            prec = [v for k, v in precision_rets[dataset_name][tracker_name].items()]
            mean_prec[i, j] = np.mean(prec, axis=0)[20]

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
        small_columns += " & AUC & DP"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers)):
        if i == nexperts:
            latex += "\\hdashline\n"

        if i >= nexperts:
            line = trackers[i][:trackers[i].index("_")]
        else:
            line = trackers[i].replace("_", "\\_")
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

    print(latex)

    preview(
        latex,
        viewer="file",
        filename=eval_dir / "table.png",
        packages=("multirow", "xcolor", "arydshln"),
    )


def draw_ranks(datasets, trackers, rets, eval_dir):
    for dataset_name in datasets:
        seq_names = sorted(rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(rets, trackers, dataset_name, seq_names)
        fig = go.Figure()
        for i in range(len(trackers)):
            fig.add_trace(
                go.Scatter(
                    x=seq_names,
                    y=ranks[:, i],
                    mode="lines+markers",
                    name=f"{trackers[i]}[{np.mean(ranks[:,i]):.2f}]"
                    if "AAA" not in trackers[i]
                    else f"AAA[{np.mean(ranks[:,i]):.2f}]",
                )
            )
        fig.update_yaxes(autorange="reversed")
        pio.write_html(fig, str(eval_dir / f"{dataset_name}.html"))


def draw_rank_radar(datasets, trackers, rets, eval_dir):
    for dataset_name in datasets:
        seq_names = sorted(rets[dataset_name][trackers[0]].keys())
        ranks = calc_rank(rets, trackers, dataset_name, seq_names)

        sorted_idx = []
        for i, tracker_name in enumerate(trackers):
            idx = np.where(ranks[:, i] == 1)[0]
            sorted_idx += idx.tolist()

        fig = go.Figure()
        for i, tracker_name in enumerate(trackers):
            fig.add_trace(
                go.Scatterpolar(
                    r=ranks[sorted_idx, i], fill="toself", name=tracker_name
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[len(trackers), 1], showticklabels=False, ticks=""
                )
            )
        )
        pio.write_html(fig, str(eval_dir / f"{dataset_name}-Rank.html"))


def main(experts, baselines, algorithms, eval_dir):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets_name = ["OTB", "NFS", "UAV", "TPL", "VOT", "LaSOT"]
    datasets = [otb, nfs, uav, tpl, vot, lasot]
    eval_trackers = experts + baselines + algorithms
    viz_trackers = experts + algorithms

    colors = sns.color_palette("hls", len(viz_trackers) + 1).as_hex()[::-1][1:]
    sns.set_palette(colors)

    eval_save = eval_dir / "eval.pkl"
    if eval_save.exists():
        successes, precisions, anchor_frames = pickle.loads(eval_save.read_bytes())
    else:
        successes = {}
        precisions = {}
        anchor_frames = {}

        for dataset, name in zip(datasets, datasets_name):
            benchmark = OPEBenchmark(dataset)

            success = benchmark.eval_success(eval_trackers)
            precision = benchmark.eval_precision(eval_trackers)
            anchor_frame = benchmark.eval_anchor_frame(algorithms)
            successes[name] = success
            precisions[name] = precision
            anchor_frames[name] = anchor_frame
        eval_save.write_bytes(pickle.dumps((successes, precisions, anchor_frames)))

    # make_table(
    #     datasets_name, eval_trackers, len(experts), successes, precisions, eval_dir
    # )

    # figsize = (20, 6)
    # draw_curves(datasets_name, viz_trackers, successes, precisions, figsize, eval_dir)
    # draw_dists(datasets_name, viz_trackers, successes, precisions, figsize, eval_dir)
    # draw_scores(datasets_name, viz_trackers, successes, precisions, figsize, eval_dir)
    # draw_score_hists(
    #     datasets_name, viz_trackers, successes, precisions, figsize, eval_dir
    # )
    # draw_rank_hists(datasets_name, viz_trackers, successes, figsize, eval_dir)

    # figsize = (20, 5)
    # draw_rank_hists(datasets_name, viz_trackers, successes, figsize, eval_dir)
    # draw_rank_sum(datasets_name, viz_trackers, successes, figsize, eval_dir)

    figsize = (10, 5)
    draw_ratio_rank(
        datasets_name,
        viz_trackers,
        successes,
        precisions,
        anchor_frames,
        figsize,
        eval_dir,
    )
    # for algorithm in algorithms:
    #     draw_ratio(
    #         datasets_name,
    #         algorithm,
    #         successes,
    #         precisions,
    #         anchor_frames,
    #         figsize,
    #         eval_dir,
    #     )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithms", default=list(), nargs="+")
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-b", "--baselines", default=list(), nargs="+")
    parser.add_argument("-d", "--dir", default="Expert", type=str)
    args = parser.parse_args()

    eval_dir = Path(f"./evaluation_results/{args.dir}")
    os.makedirs(eval_dir, exist_ok=True)

    main(args.experts, args.baselines, args.algorithms, eval_dir)
