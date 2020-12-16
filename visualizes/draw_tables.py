import numpy as np
from sympy import preview


def make_score_table(
    datasets,
    algorithms_name,
    experts_name,
    success_rets,
    precision_rets,
    save_dir,
    filename=None,
    drop_dp=False,
    drop_last_dp=False
):
    trackers_name = experts_name + algorithms_name
    mean_succ = np.zeros((len(trackers_name), len(datasets)))
    mean_prec = np.zeros((len(trackers_name), len(datasets)))
    for i, tracker_name in enumerate(trackers_name):
        for j, dataset_name in enumerate(datasets):
            succ = [
                v
                for v in success_rets[dataset_name][tracker_name].values()
                if not np.any(np.isnan(v))
            ]
            mean_succ[i, j] = np.mean(succ)

            if not drop_dp:
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
    if drop_dp:
        num_header = len(datasets)
    elif drop_last_dp:
        num_header = len(datasets) * 2 - 1
    else:
        num_header = len(datasets) * 2
    for i in range(num_header):
        header += "|c"

    latex += f"\\begin{{tabular}}{{{header}}}\n"
    latex += "\\hline\n"

    columns = "\\multirow{2}{*}{Tracker}"

    for i in range(len(datasets)):
        if drop_dp or (drop_last_dp and i == len(datasets) - 1):
            columns += f" & \\multicolumn{{1}}{{c|}}{{{datasets[i]}}}"
        else:
            columns += f" & \\multicolumn{{2}}{{c|}}{{{datasets[i]}}}"
    latex += f"{columns} \\\\\n"

    small_columns = " "
    for i in range(len(datasets)):
        if drop_dp or (drop_last_dp and i == len(datasets) - 1):
            small_columns += " & AUC"
        else:
            small_columns += " & AUC & DP"
    latex += f"{small_columns} \\\\\n"
    latex += "\\hline\\hline\n"

    for i in range(len(trackers_name)):
        if i == experts_name:
            latex += "\\hdashline\n"

        if (i >= len(experts_name)) and ("_" in trackers_name[i]):
            line = trackers_name[i][: trackers_name[i].index("_")]
        else:
            line = trackers_name[i].replace("_", "\\_")

        for j in range(len(datasets)):
            if drop_dp or (drop_last_dp and i == len(datasets) - 1):
                values = [mean_succ]
            else:
                values = [mean_succ, mean_prec]
            for value in values:
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
