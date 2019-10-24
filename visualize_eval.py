from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from evaluations.ope_benchmark import OPEBenchmark

sns.set()
sns.set_style("whitegrid")


def draw_curves(datasets, trackers, success_rets, precision_rets, figsize, eval_dir):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))

    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), sharey=True, figsize=figsize)
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
                thresholds, np.mean(value, axis=0), label=tracker_name, linewidth=2
            )[0]

            if i == 0:
                lines.append(line)

        if i == 0:
            ax.set_ylabel("Success")

    for i, dataset_name in enumerate(datasets):
        # draw precision plot
        ax = axes[1, i]

        # draw curve of trackers
        thresholds = np.arange(0, 51, 1)
        for tracker_name in trackers:
            value = [v for k, v in precision_rets[dataset_name][tracker_name].items()]
            ax.plot(thresholds, np.mean(value, axis=0), label=tracker_name, linewidth=2)

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

    fig.legend(lines, trackers, loc="upper center", ncol=(len(trackers) + 1) // 2)

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(eval_dir / "curves.pdf", bbox_inches="tight")
    plt.close()


def draw_scores(datasets, trackers, success_rets, precision_rets, figsize, eval_dir):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))
    ind = np.arange(len(datasets)) * 2
    width = 0.17

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharey=True, sharex=True, figsize=figsize
    )
    fig.add_subplot(111, frameon=False)

    lines = []

    # draw Success score
    ax = axes[0]
    succs = {}
    for tracker_name in trackers:
        values = []
        for dataset in datasets:
            value = [v for k, v in success_rets[dataset][tracker_name].items()]
            values.append(np.mean(value))
        succs[tracker_name] = values
    for idx, tracker_name in enumerate(trackers):
        value = succs[tracker_name]
        line = ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            value,
            width,
            label=tracker_name,
        )
        lines.append(line)
    ax.set_ylabel("AUC of Success")
    ax.set_xticks(ind)

    # draw Precision score
    ax = axes[1]
    precs = {}
    for tracker_name in trackers:
        values = []
        for dataset in datasets:
            value = [v for k, v in precision_rets[dataset][tracker_name].items()]
            values.append(np.mean(value, axis=0)[20])
        precs[tracker_name] = values
    for idx, tracker_name in enumerate(trackers):
        value = precs[tracker_name]
        ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            value,
            width,
            label=tracker_name,
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

    fig.legend(lines, trackers, loc="upper center", ncol=(len(trackers) + 1) // 2)

    plt.savefig(eval_dir / "scores.pdf", bbox_inches="tight")
    plt.close()


def draw_dists(datasets, trackers, success_rets, precision_rets, figsize, eval_dir):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))

    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), sharey=True, figsize=figsize)
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
                label=tracker_name if i == 0 else None,
            )

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


def draw_ratio(
    datasets, algorithm, success_rets, precision_rets, anchor_frames, figsize, eval_dir
):
    colors = sns.color_palette("hls", len(success_rets.keys()) + 1).as_hex()

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
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
        line = ax.scatter(ratio, succs, c=colors[i], label=dataset)
        lines.append(line)
    ax.set_ylabel("AUC of Success")

    ax = axes[0]
    for i, dataset in enumerate(datasets):
        precs = []
        ratio = []
        for seq in success_rets[dataset][algorithm].keys():
            precs.append(precision_rets[dataset][algorithm][seq][20])
            anchor_frame = anchor_frames[dataset][algorithm][seq]
            ratio.append(sum(anchor_frame) / len(anchor_frame))
        ax.scatter(ratio, precs, c=colors[i], label=dataset)
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

    plt.savefig(eval_dir / "ratios.pdf", bbox_inches="tight")
    plt.close()


def main(experts, algorithm, eval_dir):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets_name = ["OTB", "NFS", "UAV", "TPL", "VOT", "LaSOT"]
    datasets = [otb, nfs, uav, tpl, vot, lasot]
    trackers = experts + [algorithm]

    eval_save = eval_dir / "eval.pkl"
    if eval_save.exists():
        successes, precisions, anchor_frames = pickle.loads(eval_save.read_bytes())
    else:
        successes = {}
        precisions = {}
        anchor_frames = {}

        for dataset, name in zip(datasets, datasets_name):
            benchmark = OPEBenchmark(dataset)

            success = benchmark.eval_success(trackers)
            precision = benchmark.eval_precision(trackers)
            anchor_frame = benchmark.eval_anchor_frame([algorithm])
            successes[name] = success
            precisions[name] = precision
            anchor_frames[name] = anchor_frame
        eval_save.write_bytes(pickle.dumps((successes, precisions, anchor_frames)))

    figsize = (20, 8)
    draw_curves(datasets_name, trackers, successes, precisions, figsize, eval_dir)
    draw_dists(datasets_name, trackers, successes, precisions, figsize, eval_dir)
    draw_scores(datasets_name, trackers, successes, precisions, figsize, eval_dir)

    figsize = (10, 8)
    draw_ratio(datasets_name, algorithm, successes, precisions, anchor_frames, figsize, eval_dir)


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

    eval_dir = Path("./Eval")

    main(experts, "AAA", eval_dir)
