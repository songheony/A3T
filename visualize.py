import os
from pathlib import Path
import numpy as np
import seaborn as sns

from select_options import select_datasets
from evaluations.eval_trackers import evaluate
from evaluations.ope_benchmark import OPEBenchmark
from visualizes.draw_figures import (
    draw_pie,
    draw_result,
    draw_graph,
    draw_curves,
    draw_rank,
    draw_succ_with_thresholds,
)
from visualizes.draw_tables import (
    get_mean_succ,
    get_mean_prec,
    get_mean_fps,
    get_mean_ratio,
    make_score_table,
    make_rank_table,
)
import path_config


def get_parameter_results(eval_dir, algorithm_name, modes):
    dataset_name = "OTB2015"
    dataset = select_datasets(dataset_name)
    ope = OPEBenchmark(dataset, dataset_name)

    thresholds = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]

    threshold_successes = {mode: {} for mode in modes}
    threshold_anchor_successes = {mode: {} for mode in modes}
    threshold_anchors = {mode: {} for mode in modes}

    for mode in modes:
        for threshold in thresholds:
            parameter_name = f"{threshold:.2f}"
            if parameter_name in threshold_successes[mode].keys():
                continue

            name = f"{algorithm_name}/{mode}/{parameter_name}"

            (
                tracking_time_rets,
                success_rets,
                precision_rets,
                norm_precision_rets,
                anchor_success_rets,
                anchor_precision_rets,
                anchor_norm_precision_rets,
                error_rets,
                loss_rets,
                offline_success_rets,
                offline_precision_rets,
                anchor_frame_rets,
            ) = evaluate([dataset], [dataset_name], [], [], name,)

            threshold_successes[mode][parameter_name] = success_rets[dataset_name][name]
            threshold_anchor_successes[mode][parameter_name] = anchor_success_rets[
                dataset_name
            ][name]
            threshold_anchors[mode][parameter_name] = anchor_frame_rets[dataset_name]

    gt_trajs = ope.get_gt_trajs()

    return (
        threshold_successes,
        threshold_anchor_successes,
        threshold_anchors,
        gt_trajs,
    )


def figure1(
    datasets_name, all_experts, all_experts_name, all_successes, color_map, save_dir
):
    figsize = (10, 5)
    colors = [color_map[tracker_name] for tracker_name in all_experts]
    sns.set_palette(colors)
    draw_pie(
        datasets_name,
        all_experts,
        all_experts_name,
        all_successes,
        figsize,
        save_dir,
        legend=True,
        file_name="Figure1",
    )


def figure2(vot_dataset, all_experts, color_map, save_dir):
    target_seqs = ["fish1", "nature"]
    draw_result(
        vot_dataset,
        "VOT2018",
        None,
        all_experts,
        color_map,
        save_dir / "Figure2",
        target_seqs,
        show=["frame"],
        vis_gt=True,
        vis_best=True,
    )


def figure4(
    datasets_name,
    superfast_algorithm,
    superfast_experts,
    superfast_successes,
    superfast_precisions,
    color_map,
    save_dir,
):
    figsize = (20, 5)
    draw_curves(
        datasets_name,
        superfast_algorithm,
        superfast_experts,
        color_map,
        superfast_successes,
        superfast_precisions,
        figsize,
        save_dir,
        file_name="Figure4",
    )


def figure5(
    datasets_name,
    superfast_algorithm,
    superfast_experts,
    superfast_successes,
    fast_algorithm,
    fast_experts,
    fast_successes,
    normal_algorithm,
    normal_experts,
    normal_successes,
    color_map,
    save_dir,
):
    figsize = (20, 7)
    superfast_trackers = superfast_experts + [superfast_algorithm]
    fast_trackers = fast_experts + [fast_algorithm]
    normal_trackers = normal_experts + [normal_algorithm]
    draw_rank(
        datasets_name,
        ["SuperFast", "Fast", "Normal"],
        [superfast_trackers, fast_trackers, normal_trackers],
        [superfast_successes, fast_successes, normal_successes],
        color_map,
        figsize,
        save_dir,
        legend=True,
        file_name="Figure5",
    )


def figure6(
    otb_dataset, tpl_dataset, superfast_algorithm, superfast_experts, color_map, save_dir
):
    otb_seqs = ["Girl2"]
    draw_graph(
        otb_dataset,
        "OTB2015",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure6",
        otb_seqs,
        iserror=True,
        legend=True,
        sframes=[
            (1, "(a)"),
            (38, "(b)"),
            (137, "(c)"),
            (233, "(d)"),
            (352, "(e)"),
            (633, "(f)"),
            (809, "(g)"),
            (1239, "(h)"),
        ],
    )
    draw_graph(
        otb_dataset,
        "OTB2015",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure6",
        otb_seqs,
        iserror=False,
        legend=False,
        sframes=[
            (1, "(a)"),
            (38, "(b)"),
            (137, "(c)"),
            (233, "(d)"),
            (352, "(e)"),
            (633, "(f)"),
            (809, "(g)"),
            (1239, "(h)"),
        ],
    )
    draw_result(
        otb_dataset,
        "OTB2015",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure6",
        otb_seqs,
        show=["frame"],
    )

    tpl_seqs = ['tpl_Ball_ce2']
    draw_graph(
        tpl_dataset,
        "TColor128",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure6",
        tpl_seqs,
        iserror=True,
        legend=False,
        sframes=[
            (1, "(i)"),
            (254, "(j)"),
            (275, "(k)"),
            (329, "(l)"),
            (342, "(m)"),
            (416, "(n)"),
            (494, "(o)"),
            (571, "(p)"),
        ],
    )
    draw_graph(
        tpl_dataset,
        "TColor128",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure6",
        tpl_seqs,
        iserror=False,
        legend=False,
        sframes=[
            (1, "(i)"),
            (254, "(j)"),
            (275, "(k)"),
            (329, "(l)"),
            (342, "(m)"),
            (416, "(n)"),
            (494, "(o)"),
            (571, "(p)"),
        ],
    )
    draw_result(
        tpl_dataset,
        "TColor128",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure6",
        tpl_seqs,
        show=["frame"],
    )


def figure7(thresholds, threshold_successes, threshold_anchor_successes, threshold_anchors, gt_trajs, save_dir):
    figsize = (15, 5)
    modes = threshold_successes.keys()
    draw_succ_with_thresholds(
        modes,
        thresholds,
        threshold_successes,
        threshold_anchor_successes,
        threshold_anchors,
        gt_trajs,
        figsize,
        save_dir,
        "Figure7",
    )


def figure8(
    otb_dataset, superfast_algorithm, superfast_mcct_algorithm, superfast_experts, color_map, save_dir
):
    target_seqs = ["Bird1", "Tiger1"]
    draw_result(
        otb_dataset,
        "OTB2015",
        superfast_mcct_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure8" / "MCCT",
        target_seqs,
        show=["frame"],
    )

    draw_result(
        otb_dataset,
        "OTB2015",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure8" / "Ours",
        target_seqs,
        show=["frame"],
    )


def figure9(
    otb_dataset, superfast_algorithm, superfast_hdt_algorithm, superfast_experts, color_map, save_dir
):
    target_seqs = ['Human4_2', 'Skating1']
    draw_result(
        otb_dataset,
        "OTB2015",
        superfast_hdt_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure9" / "HDT",
        target_seqs,
        show=["frame"],
    )

    draw_result(
        otb_dataset,
        "OTB2015",
        superfast_algorithm,
        superfast_experts,
        color_map,
        save_dir / "Figure9" / "Ours",
        target_seqs,
        show=["frame"],
    )


def score_table(
    datasets_name,
    algorithm,
    baselines,
    experts,
    successes,
    precisions,
    tracking_times,
    save_dir,
    filename,
    metrics=["AUC", "DP"],
    with_best=True,
):

    algorithms = [algorithm]
    if with_best:
        for dataset_name in datasets_name:
            seq_names = sorted(successes[dataset_name][algorithm].keys())
            successes[dataset_name]["Best expert"] = dict()
            precisions[dataset_name]["Best expert"] = dict()
            tracking_times[dataset_name]["Best expert"] = dict()
            for seq_name in seq_names:
                max_score = 0
                for expert_name in experts:
                    score = np.mean(successes[dataset_name][expert_name][seq_name])
                    if max_score < score:
                        successes[dataset_name]["Best expert"][seq_name] = score
                        precisions[dataset_name]["Best expert"][seq_name] = precisions[dataset_name][expert_name][seq_name]
                        tracking_times[dataset_name]["Best expert"][seq_name] = tracking_times[dataset_name][expert_name][seq_name]
                        max_score = score

        algorithms += ["Best expert"]

    trackers = experts + baselines + algorithms

    mean_values = {}
    if "AUC" in metrics:
        mean_values["AUC"] = get_mean_succ(trackers, datasets_name, successes).round(3)
    if "DP" in metrics:
        mean_values["DP"] = get_mean_prec(trackers, datasets_name, precisions).round(3)
    if "FPS" in metrics:
        mean_values["FPS"] = (
            get_mean_fps(trackers, datasets_name, tracking_times).round(0).astype(int)
        )

    make_score_table(
        datasets_name,
        baselines + algorithms,
        experts,
        mean_values,
        save_dir,
        filename=filename,
    )


def threshold_table(
    algorithm_name,
    thresholds,
    threshold_successes,
    threshold_anchor_successes,
    threshold_anchors,
    gt_trajs,
    save_dir,
):
    modes = list(threshold_successes.keys())
    mean_values = {
        "AUC": get_mean_succ(thresholds, modes, threshold_successes).round(3),
        "Anchor": get_mean_succ(thresholds, modes, threshold_anchor_successes).round(3),
        "Ratio": get_mean_ratio(thresholds, modes, threshold_anchors, gt_trajs).round(
            2
        ),
    }
    make_score_table(
        modes,
        thresholds,
        [],
        mean_values,
        save_dir,
        filename=f"Threshold_{algorithm_name}",
    )


def main(experiments, all_experts, all_experts_name):
    save_dir = Path(path_config.VISUALIZATION_PATH)
    os.makedirs(save_dir, exist_ok=True)

    total_colors = sns.color_palette("hls", len(all_experts) + 2).as_hex()
    color_map = {
        tracker_name: color
        for tracker_name, color in zip(all_experts, total_colors[1:])
    }
    color_map["MCCT"] = total_colors[0]
    color_map["HDT"] = total_colors[0]
    color_map["Random"] = total_colors[0]
    color_map["WithoutOffline"] = total_colors[0]
    color_map["WithoutDelay"] = total_colors[0]
    color_map["A3T"] = total_colors[0]

    datasets_name = [
        "OTB2015",
        "TColor128",
        "UAV123",
        "NFS",
        "LaSOT",
        "VOT2018",
    ]
    datasets = [select_datasets(dataset_name) for dataset_name in datasets_name]

    # All
    (
        tracking_time_rets,
        success_rets,
        precision_rets,
        norm_precision_rets,
        anchor_success_rets,
        anchor_precision_rets,
        anchor_norm_precision_rets,
        error_rets,
        loss_rets,
        offline_success_rets,
        offline_precision_rets,
        anchor_frame_rets,
    ) = evaluate(datasets, datasets_name, all_experts, [], None)
    figure1(
        datasets_name, all_experts, all_experts_name, success_rets, color_map, save_dir
    )

    # Figure 2
    figure2(datasets[5], all_experts, color_map, save_dir)

    # Super Fast
    superfast_algorithm, superfast_baselines, superfast_experts = experiments[
        "SuperFast"
    ]
    (
        superfast_tracking_time_rets,
        superfast_success_rets,
        superfast_precision_rets,
        superfast_norm_precision_rets,
        superfast_anchor_success_rets,
        superfast_anchor_precision_rets,
        superfast_anchor_norm_precision_rets,
        superfast_error_rets,
        superfast_loss_rets,
        superfast_offline_success_rets,
        superfast_offline_precision_rets,
        superfast_anchor_frame_rets,
    ) = evaluate(
        datasets,
        datasets_name,
        superfast_experts,
        superfast_baselines,
        superfast_algorithm,
    )

    # Table1
    score_table(
        datasets_name,
        superfast_algorithm,
        superfast_baselines,
        superfast_experts,
        superfast_success_rets,
        superfast_precision_rets,
        superfast_tracking_time_rets,
        save_dir,
        filename="Table1",
    )

    # Table1_FPS
    score_table(
        datasets_name,
        superfast_algorithm,
        superfast_baselines,
        superfast_experts,
        superfast_success_rets,
        superfast_precision_rets,
        superfast_tracking_time_rets,
        save_dir,
        filename="Table1_FPS",
        metrics=["FPS"],
        with_best=False,
    )

    # Table6
    score_table(
        datasets_name,
        superfast_algorithm,
        [],
        superfast_experts,
        superfast_anchor_success_rets,
        None,
        None,
        save_dir,
        filename="Table6",
        metrics=["AUC"],
        with_best=False,
    )

    # Table 9
    make_rank_table(
        datasets_name, superfast_experts, superfast_error_rets, superfast_loss_rets, save_dir, "Table9"
    )

    figure4(
        datasets_name,
        superfast_algorithm,
        superfast_experts,
        superfast_success_rets,
        superfast_precision_rets,
        color_map,
        save_dir,
    )

    figure6(datasets[0], datasets[1], superfast_algorithm, superfast_experts, color_map, save_dir)
    figure8(datasets[0], superfast_algorithm, superfast_baselines[2], superfast_experts, color_map, save_dir)
    figure9(datasets[0], superfast_algorithm, superfast_baselines[1], superfast_experts, color_map, save_dir)

    # Fast
    fast_algorithm, fast_baselines, fast_experts = experiments["Fast"]
    (
        fast_tracking_time_rets,
        fast_success_rets,
        fast_precision_rets,
        fast_norm_precision_rets,
        fast_anchor_success_rets,
        fast_anchor_precision_rets,
        fast_anchor_norm_precision_rets,
        fast_error_rets,
        fast_loss_rets,
        fast_offline_success_rets,
        fast_offline_precision_rets,
        fast_anchor_frame_rets,
    ) = evaluate(datasets, datasets_name, fast_experts, fast_baselines, fast_algorithm)

    # Table2
    score_table(
        datasets_name,
        fast_algorithm,
        fast_baselines,
        fast_experts,
        fast_success_rets,
        fast_precision_rets,
        fast_tracking_time_rets,
        save_dir,
        filename="Table2",
    )

    # Table2 FPS
    score_table(
        datasets_name,
        fast_algorithm,
        fast_baselines,
        fast_experts,
        fast_success_rets,
        fast_precision_rets,
        fast_tracking_time_rets,
        save_dir,
        filename="Table2_FPS",
        metrics=["FPS"],
        with_best=False,
    )

    # Table7
    score_table(
        datasets_name,
        fast_algorithm,
        [],
        fast_experts,
        fast_anchor_success_rets,
        None,
        None,
        save_dir,
        filename="Table7",
        metrics=["AUC"],
        with_best=False,
    )

    # Table 10
    make_rank_table(
        datasets_name, fast_experts, fast_error_rets, fast_loss_rets, save_dir, "Table10"
    )

    # Normal
    normal_algorithm, normal_baselines, normal_experts = experiments["Normal"]
    (
        normal_tracking_time_rets,
        normal_success_rets,
        normal_precision_rets,
        normal_norm_precision_rets,
        normal_anchor_success_rets,
        normal_anchor_precision_rets,
        normal_anchor_norm_precision_rets,
        normal_error_rets,
        normal_loss_rets,
        normal_offline_success_rets,
        normal_offline_precision_rets,
        normal_anchor_frame_rets,
    ) = evaluate(
        datasets, datasets_name, normal_experts, normal_baselines, normal_algorithm
    )

    # Table3
    score_table(
        datasets_name,
        normal_algorithm,
        normal_baselines,
        normal_experts,
        normal_success_rets,
        normal_precision_rets,
        normal_tracking_time_rets,
        save_dir,
        filename="Table3",
    )

    # Table3 FPS
    score_table(
        datasets_name,
        normal_algorithm,
        normal_baselines,
        normal_experts,
        normal_success_rets,
        normal_precision_rets,
        normal_tracking_time_rets,
        save_dir,
        filename="Table3_FPS",
        metrics=["FPS"],
        with_best=False,
    )

    # Table8
    score_table(
        datasets_name,
        normal_algorithm,
        [],
        normal_experts,
        normal_anchor_success_rets,
        None,
        None,
        save_dir,
        filename="Table8",
        metrics=["AUC"],
        with_best=False,
    )

    # Table 11
    make_rank_table(
        datasets_name, normal_experts, normal_error_rets, normal_loss_rets, save_dir, "Table11"
    )

    # Figure5
    figure5(
        datasets_name,
        superfast_algorithm,
        superfast_experts,
        superfast_success_rets,
        fast_algorithm,
        fast_experts,
        fast_success_rets,
        normal_algorithm,
        normal_experts,
        normal_success_rets,
        color_map,
        save_dir,
    )

    # SiamDW
    siamdw_algorithm, siamdw_baselines, siamdw_experts = experiments["SiamDW"]
    (
        siamdw_tracking_time_rets,
        siamdw_success_rets,
        siamdw_precision_rets,
        siamdw_norm_precision_rets,
        siamdw_anchor_success_rets,
        siamdw_anchor_precision_rets,
        siamdw_anchor_norm_precision_rets,
        siamdw_error_rets,
        siamdw_loss_rets,
        siamdw_offline_success_rets,
        siamdw_offline_precision_rets,
        siamdw_anchor_frame_rets,
    ) = evaluate(
        datasets, datasets_name, siamdw_experts, siamdw_baselines, siamdw_algorithm
    )

    # Table4
    score_table(
        datasets_name,
        siamdw_algorithm,
        siamdw_baselines,
        siamdw_experts,
        siamdw_success_rets,
        siamdw_precision_rets,
        siamdw_tracking_time_rets,
        save_dir,
        filename="Table4",
    )

    # Table4 FPS
    score_table(
        datasets_name,
        siamdw_algorithm,
        siamdw_baselines,
        siamdw_experts,
        siamdw_success_rets,
        siamdw_precision_rets,
        siamdw_tracking_time_rets,
        save_dir,
        filename="Table4_FPS",
        metrics=["FPS"],
        with_best=False,
    )

    # SiamRPN++
    siamrpn_algorithm, siamrpn_baselines, siamrpn_experts = experiments["SiamRPN++"]
    (
        siamrpn_tracking_time_rets,
        siamrpn_success_rets,
        siamrpn_precision_rets,
        siamrpn_norm_precision_rets,
        siamrpn_anchor_success_rets,
        siamrpn_anchor_precision_rets,
        siamrpn_anchor_norm_precision_rets,
        siamrpn_error_rets,
        siamrpn_loss_rets,
        siamrpn_offline_success_rets,
        siamrpn_offline_precision_rets,
        siamrpn_anchor_frame_rets,
    ) = evaluate(
        datasets, datasets_name, siamrpn_experts, siamrpn_baselines, siamrpn_algorithm
    )

    # Table5
    score_table(
        datasets_name,
        siamrpn_algorithm,
        siamrpn_baselines,
        siamrpn_experts,
        siamrpn_success_rets,
        siamrpn_precision_rets,
        siamrpn_tracking_time_rets,
        save_dir,
        filename="Table5",
    )

    # Table5 FPS
    score_table(
        datasets_name,
        siamrpn_algorithm,
        siamrpn_baselines,
        siamrpn_experts,
        siamrpn_success_rets,
        siamrpn_precision_rets,
        siamrpn_tracking_time_rets,
        save_dir,
        filename="Table5_FPS",
        metrics=["FPS"],
        with_best=False,
    )

    # Parameters
    target_modes = ["SuperFast", "Fast", "Normal", "SiamDW", "SiamRPN++"]
    (
        threshold_successes, threshold_anchor_successes, threshold_anchors, gt_trajs
    ) = get_parameter_results(path_config.EVALUATION_PATH, "A3T", target_modes)
    thresholds = list(threshold_successes[target_modes[0]].keys())
    figure7(thresholds, threshold_successes, threshold_anchor_successes, threshold_anchors, gt_trajs, save_dir)


if __name__ == "__main__":
    all_experts = [
        "DaSiamRPN",
        "DiMP",
        "DROL",
        "KYS",
        "Ocean",
        "PrDiMP",
        "RPT",
        "SiamBAN",
        "SiamCAR",
        "SiamDW",
        "SiamFC++",
        "SiamRPN",
        "SiamRPN++",
        "SPM",
    ]
    all_experts_name = [
        "DaSiamRPN",
        "DiMP",
        "DROL",
        "KYS",
        "Ocean",
        "PrDiMP",
        "RPT",
        "SiamBAN",
        "SiamCAR",
        "SiamDW",
        "SiamFC++",
        "SiamRPN",
        "SiamRPN++",
        "SPM",
    ]

    super_fast_algorithm = "A3T/SuperFast/0.70"
    fast_algorithm = "A3T/Fast/0.72"
    normal_algorithm = "A3T/Normal/0.74"
    siamdw_algorithm = "A3T/SiamDW/0.73"
    siamrpn_algorithm = "A3T/SiamRPN++/0.75"

    super_fast_baselines = [
        "Random/SuperFast",
        "HDT/SuperFast/0.65",
        "MCCT/SuperFast/0.10",
        "WithoutDelay/SuperFast",
    ]
    fast_baselines = [
        "Random/Fast",
        "HDT/Fast/0.65",
        "MCCT/Fast/0.10",
        "WithoutDelay/Fast",
    ]
    normal_baselines = [
        "Random/Normal",
        "HDT/Normal/0.65",
        "MCCT/Normal/0.10",
        "WithoutDelay/Normal",
    ]
    siamdw_baselines = [
        "Random/SiamDW",
        "HDT/SiamDW/0.65",
        "MCCT/SiamDW/0.10",
        "WithoutDelay/SiamDW",
    ]
    siamrpn_baselines = [
        "Random/SiamRPN++",
        "HDT/SiamRPN++/0.65",
        "MCCT/SiamRPN++/0.10",
        "WithoutDelay/SiamRPN++",
    ]

    super_fast_experts = ["DaSiamRPN", "SiamDW", "SiamRPN", "SPM"]
    fast_experts = ["Ocean", "SiamBAN", "SiamCAR", "SiamFC++", "SiamRPN++"]
    normal_experts = ["DiMP", "DROL", "KYS", "PrDiMP", "RPT"]
    siamdw_experts = [
        "SiamDWGroup/SiamFCRes22/OTB",
        "SiamDWGroup/SiamFCIncep22/OTB",
        "SiamDWGroup/SiamFCNext22/OTB",
        "SiamDWGroup/SiamRPNRes22/OTB",
        "SiamDWGroup/SiamFCRes22/VOT",
        "SiamDWGroup/SiamFCIncep22/VOT",
        "SiamDWGroup/SiamFCNext22/VOT",
        "SiamDWGroup/SiamRPNRes22/VOT",
    ]
    siamrpn_experts = [
        "SiamRPN++Group/AlexNet/VOT",
        "SiamRPN++Group/AlexNet/OTB",
        "SiamRPN++Group/ResNet-50/VOT",
        "SiamRPN++Group/ResNet-50/OTB",
        "SiamRPN++Group/ResNet-50/VOTLT",
        "SiamRPN++Group/MobileNetV2/VOT",
        "SiamRPN++Group/SiamMask/VOT",
    ]

    experiments = {
        "SuperFast": (super_fast_algorithm, super_fast_baselines, super_fast_experts),
        "Fast": (fast_algorithm, fast_baselines, fast_experts),
        "Normal": (normal_algorithm, normal_baselines, normal_experts),
        "SiamDW": (siamdw_algorithm, siamdw_baselines, siamdw_experts),
        "SiamRPN++": (siamrpn_algorithm, siamrpn_baselines, siamrpn_experts),
    }

    main(experiments, all_experts, all_experts_name)
