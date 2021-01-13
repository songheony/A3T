import numpy as np
from sympy import preview


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
    metrics_name = sorted(mean_values.keys())

    latex = "\\begin{table*}\n"
    latex += "\\centering\n"
    latex += "\\begin{threeparttable}\n"

    header = "c"
    num_header = len(datasets_name) * len(metrics_name)
    for i in range(num_header):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Tracker}"

    for i in range(len(datasets_name)):
        dataset_name = datasets_name[i].replace("%", "\\%")
        small_colunm = "c|" if i < len(datasets_name) - 1 else "c"
        columns += f" & \\multicolumn{{{len(metrics_name)}}}{{{small_colunm}}}{{{dataset_name}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(datasets_name)):
        for j in range(len(metrics_name)):
            small_columns += f" & {metrics_name[j]}"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers_name)):
        if i == len(experts_name):
            latex += "\\hdashline\n"

        line = trackers_name[i]
        for j in range(len(datasets_name)):
            for metric_name in metrics_name:
                value = mean_values[metric_name]
                sorted_idx = np.argsort(value[:, j])
                if i == sorted_idx[-1]:
                    line += f" & {{\\color{{red}} \\textbf{{{value[i, j]:0.2f}}}}}"
                elif i == sorted_idx[-2]:
                    line += f" & {{\\color{{blue}} \\textit{{{value[i, j]:0.2f}}}}}"
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
    txt_file = save_dir / f"{filename}.txt"
    txt_file.write_text(latex)

    preview(
        latex,
        viewer="file",
        filename=save_dir / f"{filename}.png",
        packages=("multirow", "xcolor", "arydshln", "threeparttable"),
    )
