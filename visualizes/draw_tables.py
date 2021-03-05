import numpy as np
from sympy import preview


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def is_algorithm(tracker_name):
    for algorithm in ["HDT", "MCCT", "Random", "WithoutDelay", "AAA"]:
        if tracker_name.startswith(algorithm):
            return True
    return False


def calc_rank(dataset_name, seq_names, trackers, rets, mean=True):
    ranks = []
    for seq_name in seq_names:
        if mean:
            value = np.mean(
                [
                    rets[dataset_name][tracker_name][seq_name]
                    for tracker_name in trackers
                ],
                axis=1,
            )
        else:
            value = np.array(
                [
                    rets[dataset_name][tracker_name][seq_name]
                    for tracker_name in trackers
                ]
            )
        temp = value.argsort()
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(value))
        rank = len(trackers) - rank
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks


def get_mean_succ(trackers_name, datasets_name, success_rets):
    mean_succ = np.zeros((len(trackers_name), len(datasets_name)))
    for i, tracker_name in enumerate(trackers_name):
        for j, dataset_name in enumerate(datasets_name):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)

    return mean_succ


def get_mean_prec(trackers_name, datasets_name, precision_rets):
    mean_prec = np.zeros((len(trackers_name), len(datasets_name)))
    for i, tracker_name in enumerate(trackers_name):
        for j, dataset_name in enumerate(datasets_name):
            prec = [
                v
                for v in precision_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_prec[i, j] = np.mean(prec, axis=0)[20]

    return mean_prec


def get_mean_ratio(trackers_name, datasets_name, anchor_frames, gt_trajs):
    mean_ratio = np.zeros((len(trackers_name), len(datasets_name)))
    for i, tracker_name in enumerate(trackers_name):
        for j, dataset_name in enumerate(datasets_name):
            ratio = [
                len(v) / len(gt_trajs[k])
                for k, v in anchor_frames[dataset_name][tracker_name].items()
            ]
            mean_ratio[i, j] = np.mean(ratio)

    return mean_ratio


def get_mean_fps(trackers_name, datasets_name, tracking_time_rets):
    mean_fps = np.zeros((len(trackers_name), len(datasets_name)))
    for i, tracker_name in enumerate(trackers_name):
        for j, dataset_name in enumerate(datasets_name):
            fps = [
                1 / v for v in tracking_time_rets[dataset_name][tracker_name].values()
            ]
            mean_fps[i, j] = np.mean(fps)

    return mean_fps


def make_score_table(
    datasets_name, algorithms_name, experts_name, mean_values, save_dir, filename=None,
):
    trackers_name = experts_name + algorithms_name
    metrics_name = list(mean_values.keys())

    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c"
    num_header = len(datasets_name) * len(metrics_name)
    for i in range(num_header):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    if len(metrics_name) > 1:
        columns = "\\multirow{2}{*}{Tracker}"
    else:
        columns = "Tracker"

    for i in range(len(datasets_name)):
        dataset_name = datasets_name[i].replace("%", "\\%")
        if len(metrics_name) > 1:
            small_colunm = "c|" if i < len(datasets_name) - 1 else "c"
            columns += f" & \\multicolumn{{{len(metrics_name)}}}{{{small_colunm}}}{{{dataset_name}}}"
        else:
            columns += f" & {dataset_name}"
    latex += f"{columns} \\\\\n"

    if len(metrics_name) > 1:
        small_columns = " "
        for i in range(len(datasets_name)):
            for j in range(len(metrics_name)):
                small_columns += f" & {metrics_name[j]}"
        latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers_name)):
        if i == len(experts_name) or trackers_name[i] == "Best expert":
            latex += "\\hdashline\n"

        if is_algorithm(trackers_name[i]):
            line = trackers_name[i].split("/")[0]
            if line == "WithoutDelay":
                line = "AAA w/o delay"
        else:
            line = trackers_name[i]
            line = line.replace("Group", "")
        for j in range(len(datasets_name)):
            for metric_name in metrics_name:
                value = mean_values[metric_name]
                if metric_name == "FPS":
                    line += f" & {value[i, j]}"
                else:
                    if "Best expert" in trackers_name:
                        sorted_idx = np.argsort(value[:-1, j])
                    else:
                        sorted_idx = np.argsort(value[:, j])

                    if i == sorted_idx[-1]:
                        line += f" & {{\\color{{red}} \\textbf{{{value[i, j]}}}}}"
                    elif i == sorted_idx[-2]:
                        line += f" & {{\\color{{blue}} \\textit{{{value[i, j]}}}}}"
                    else:
                        line += f" & {value[i, j]}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table*}\n"

    if filename is None:
        filename = "table"
    txt_file = save_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=save_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln", "threeparttable"),
    )


def find_rank(
    datasets, algorithms, experts, success_rets, save_dir, filename="Ranking"
):
    text = ""
    for i, dataset_name in enumerate(datasets):
        text += f"{dataset_name}\n"
        text += "-" * 10 + "\n"
        seq_names = sorted(success_rets[dataset_name][experts[0]].keys())
        for algorithm in algorithms:
            rank = calc_rank(
                dataset_name, seq_names, experts + [algorithm], success_rets
            )[:, -1]
            first = [seq for seq, cond in zip(seq_names, rank == 1) if cond]
            last = [
                seq for seq, cond in zip(seq_names, rank == len(experts) + 1) if cond
            ]
            text += f"{algorithm.split('_')[0]}: best-{first} / worst-{last}\n"
        text += "\n"

    txt_file = save_dir / f"{filename}.txt"
    txt_file.write_text(text)


def make_rank_table(
    datasets_name, experts_name, error_rets, loss_rets, save_dir, filename=None
):
    latex = "\\begin{table}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c|c|c"
    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "Dataset & Rank & Diff"
    latex += f"{columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for dataset_name in datasets_name:
        line = dataset_name

        seq_names = sorted(error_rets[dataset_name][experts_name[0]].keys())
        error_rank = (
            len(experts_name)
            + 1
            - calc_rank(dataset_name, seq_names, experts_name, error_rets, mean=False)
        )
        loss_rank = (
            len(experts_name)
            + 1
            - calc_rank(dataset_name, seq_names, experts_name, loss_rets, mean=False)
        )

        best_expert = np.argmin(loss_rank, axis=1)
        best_expert_rank = np.take_along_axis(error_rank, best_expert[:, None], axis=1)
        line += f" & {np.mean(best_expert_rank):.2f}"

        errors = []
        for seq_name, best in zip(seq_names, best_expert):
            error = np.array(
                [
                    error_rets[dataset_name][expert_name][seq_name]
                    for expert_name in experts_name
                ]
            )
            norm_error = minmax(error)
            errors.append(norm_error[best])

        line += f" & {np.mean(errors):.2f}"
        line += " \\\\\n"

        latex += f"{line}"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{threeparttable}\n"
    latex += "\\end{table}\n"

    if filename is None:
        filename = "table"
    txt_file = save_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=save_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln", "threeparttable"),
    )
