import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("whitegrid")


def draw_curve(trackers, success_ret, precision_ret):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
    plt.grid(False)

    # draw success plot
    ax = axes[0]
    ax.set_ylabel("Success rate")
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
        [np.mean(success_ret[tracker_name].values()) for tracker_name in trackers]
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
    ax.set_ylabel("Precision rate")
    ax.set(xlim=(0, 50), ylim=(0, 1))
    thresholds = np.arange(0, 51, 1)
    for tracker_name in trackers:
        value = [v for k, v in precision_ret[tracker_name].items()]
        label = "[%.3f] %s" % (np.mean(value, axis=0)[20], tracker_name) + tracker_name
        ax.plot(thresholds, np.mean(value, axis=0), label=label, linewidth=2)

    # sort legend
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    idx = np.argsort(
        [np.mean(precision_ret[tracker_name].values()) for tracker_name in trackers]
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

    plt.savefig("curve.pdf", bbox_inches="tight")


def draw_rank(trackers, success_ret, precision_ret):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
    plt.grid(False)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")

    # draw AUC rank
    seq_performance = []
    for tracker_name in trackers:
        value = [v for k, v in success_ret[tracker_name].items()]
        seq_performance.append(np.mean(value, axis=1))
    seq_performance = np.array(seq_performance)
    ranks = np.empty_like(seq_performance)
    for seq in ranks.shape[1]:
        temp = np.argsort(seq_performance[:, seq])[::-1]
        ranks[temp, seq] = np.arange(len(temp)) + 1
    ax.hist(ranks.tolist(), label=trackers, bins=np.arange(1, len(trackers) + 2) - 0.5)

    ax.legend(labelspacing=0.2)
    ax.set_xticks(range(1, len(trackers) + 1))
    ax.set_xticklabels(["Best"] + list(range(2, len(trackers) + 1)))

    plt.savefig("rank.pdf", bbox_inches="tight")


def draw_score(datasets, trackers, success_rets):
    sns.set_palette(sns.color_palette("hls", len(trackers) + 1))
    ind = np.arange(len(datasets)) * 2
    width = 0.17

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor="none", top="off", bottom="off", left="off", right="off")
    plt.grid(False)
    plt.xlabel("Dataset")
    plt.ylabel("AUC")

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

    plt.savefig("score.pdf", bbox_inches="tight")
