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


def draw_curve(dataset_name, trackers, success_ret, precision_ret):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.add_subplot(111, frameon=False)

    # draw success plot
    ax = axes[0]
    # ax.set_ylabel("Success rate")
    ax.set_ylabel("Rate")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in trackers:
        value = [v for k, v in success_ret[tracker_name].items()]
        label = "[%.3f] %s" % (np.mean(value), tracker_name)
        ax.plot(thresholds, np.mean(value, axis=0), label=label, linewidth=2)

    # sort legend
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    idx = np.argsort(
        [np.mean(list(success_ret[tracker_name].values())) for tracker_name in trackers]
    )[::-1]
    handles, labels = [handles[i] for i in idx], [labels[i] for i in idx]
    ax.legend(handles, labels, loc="lower right", labelspacing=0.2)

    ax.autoscale(enable=True, axis="both", tight=True)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin = 0
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_xticks(np.arange(xmin, xmax + 0.01, 0.1))
    ax.set_yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin) / (ymax - ymin))

    # draw precision plot
    ax = axes[1]
    # ax.set_ylabel("Precision rate")
    ax.set(xlim=(0, 50), ylim=(0, 1))
    thresholds = np.arange(0, 51, 1)
    for tracker_name in trackers:
        value = [v for k, v in precision_ret[tracker_name].items()]
        label = "[%.3f] %s" % (np.mean(value, axis=0)[20], tracker_name)
        ax.plot(thresholds, np.mean(value, axis=0), label=label, linewidth=2)

    # sort legend
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    idx = np.argsort(
        [
            np.mean(list(precision_ret[tracker_name].values()))
            for tracker_name in trackers
        ]
    )[::-1]
    handles, labels = [handles[i] for i in idx], [labels[i] for i in idx]
    ax.legend(handles, labels, loc="lower right", labelspacing=0.2)

    ax.autoscale(enable=True, axis="both", tight=True)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin = 0
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_xticks(np.arange(xmin, xmax + 0.01, 5))
    ax.set_yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin) / (ymax - ymin))

    # hide tick and tick label of the big axes
    plt.axis('off')
    plt.grid(False)

    plt.savefig("%s_curve.pdf" % dataset_name, bbox_inches="tight")
    plt.close()


def draw_rank(dataset_name, trackers, success_ret, precision_ret):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.add_subplot(111, frameon=False)

    # draw AUC rank
    seq_performance = []
    for tracker_name in trackers:
        value = [v for k, v in success_ret[tracker_name].items()]
        seq_performance.append(np.mean(value, axis=1))
    seq_performance = np.array(seq_performance)
    ranks = np.empty_like(seq_performance)
    for seq in range(ranks.shape[1]):
        temp = np.argsort(seq_performance[:, seq])[::-1]
        ranks[temp, seq] = np.arange(len(temp)) + 1
    ax.hist(ranks.tolist(), label=trackers, bins=np.arange(1, len(trackers) + 2) - 0.5)

    ax.legend(labelspacing=0.2)
    ax.set_xticks(range(1, len(trackers) + 1))
    ax.set_xticklabels(["Best"] + list(range(2, len(trackers) + 1)))

    # hide tick and tick label of the big axes
    plt.axis('off')
    plt.grid(False)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")

    plt.savefig("%s_rank.pdf" % dataset_name, bbox_inches="tight")
    plt.close()


def draw_score(datasets, trackers, success_rets):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))
    ind = np.arange(len(datasets)) * 2
    width = 0.17

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.add_subplot(111, frameon=False)

    # draw AUC score
    aucs = {}
    for tracker_name in trackers:
        values = []
        for dataset in datasets:
            value = [v for k, v in success_rets[dataset][tracker_name].items()]
            values.append(np.mean(value))
        aucs[tracker_name] = values
    for idx, tracker_name in enumerate(trackers):
        value = aucs[tracker_name]
        ax.bar(
            ind + (idx - (len(trackers) - 1) / 2.0) * width,
            value,
            width,
            label=tracker_name,
        )
    ax.legend(labelspacing=0.2)
    ax.set_xticks(ind)
    ax.set_xticklabels(datasets)

    # hide tick and tick label of the big axes
    plt.axis('off')
    plt.grid(False)
    plt.xlabel("Dataset")
    plt.ylabel("AUC")

    plt.savefig("score.pdf", bbox_inches="tight")
    plt.close()


def main(trackers):
    otb = OTBDataset()
    nfs = NFSDataset()
    uav = UAVDataset()
    tpl = TPLDataset()
    vot = VOTDataset()
    lasot = LaSOTDataset()

    datasets = [otb, nfs, uav, tpl, vot, lasot]
    datasets_name = ["OTB", "NFS", "UAV", "TPL", "VOT", "LaSOT"]

    successes = {}
    precisions = {}

    for dataset, name in zip(datasets, datasets_name):
        benchmark = OPEBenchmark(dataset)

        success = benchmark.eval_success(trackers)
        precision = benchmark.eval_precision(trackers)
        successes[name] = success
        precisions[name] = precision

        draw_curve(name, trackers, success, precision)
        draw_rank(name, trackers, success, precision)
    draw_score(datasets_name, trackers, successes)


if __name__ == "__main__":
    trackers = [
        "ATOM",
        "DaSiamRPN",
        "ECO",
        "SiamDW",
        "SiamFC",
        "SiamRPN",
        "SiamRPN++",
        "Staple",
        "STRCF",
        "TADT",
    ]

    main(trackers)
