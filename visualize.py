import os
from pathlib import Path
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


def get_tuning_results(eval_dir, algorithm_name, modes):
    dataset_name = "Got10K"
    dataset = select_datasets(dataset_name)
    ope = OPEBenchmark(dataset, dataset_name)

    if algorithm_name == "AAA":
        thresholds = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80]
    elif algorithm_name == "HDT":
        thresholds = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

    threshold_successes = {mode: {} for mode in modes}
    threshold_precisions = {mode: {} for mode in modes}
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
                anchor_frame_rets,
            ) = evaluate([dataset], [dataset_name], [], [], name,)

            threshold_successes[mode][parameter_name] = success_rets[dataset_name][name]
            threshold_precisions[mode][parameter_name] = precision_rets[dataset_name][
                name
            ]
            threshold_anchor_successes[mode][parameter_name] = anchor_success_rets[
                dataset_name
            ][name]
            threshold_anchors[mode][parameter_name] = anchor_frame_rets[dataset_name]

    gt_trajs = ope.get_gt_trajs()

    return (
        threshold_successes,
        threshold_precisions,
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
    high_algorithm,
    high_experts,
    high_successes,
    high_precisions,
    color_map,
    save_dir,
):
    figsize = (20, 5)
    draw_curves(
        datasets_name,
        high_algorithm,
        high_experts,
        color_map,
        high_successes,
        high_precisions,
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
    otb_dataset, tpl_dataset, high_algorithm, high_experts, color_map, save_dir
):
    otb_seqs = ["Girl2"]
    draw_graph(
        otb_dataset,
        "OTB2015",
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure6",
        otb_seqs,
        iserror=True,
        legend=True,
        sframes=[
            (14, "(a)"),
            (87, "(b)"),
            (131, "(c)"),
            (150, "(d)"),
            (575, "(e)"),
            (931, "(f)"),
            (1235, "(g)"),
            (1495, "(h)"),
        ],
    )
    draw_graph(
        otb_dataset,
        "OTB2015",
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure6",
        otb_seqs,
        iserror=False,
        legend=False,
        sframes=[
            (14, "(a)"),
            (87, "(b)"),
            (131, "(c)"),
            (150, "(d)"),
            (575, "(e)"),
            (931, "(f)"),
            (1235, "(g)"),
            (1495, "(h)"),
        ],
    )
    draw_result(
        otb_dataset,
        "OTB2015",
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure6",
        otb_seqs,
        show=["frame"],
    )

    tpl_seqs = ["tpl_Yo_yos_ce1"]
    draw_graph(
        tpl_dataset,
        "TColor128",
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure6",
        tpl_seqs,
        iserror=True,
        legend=False,
        sframes=[
            (2, "(i)"),
            (29, "(j)"),
            (36, "(k)"),
            (109, "(l)"),
            (119, "(m)"),
            (175, "(n)"),
            (203, "(o)"),
            (212, "(p)"),
        ],
    )
    draw_graph(
        tpl_dataset,
        "TColor128",
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure6",
        tpl_seqs,
        iserror=False,
        legend=False,
        sframes=[
            (2, "(i)"),
            (29, "(j)"),
            (36, "(k)"),
            (109, "(l)"),
            (119, "(m)"),
            (175, "(n)"),
            (203, "(o)"),
            (212, "(p)"),
        ],
    )
    draw_result(
        tpl_dataset,
        "TColor128",
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure6",
        tpl_seqs,
        show=["frame"],
    )


def figure7(thresholds, threshold_successes, threshold_anchors, gt_trajs, save_dir):
    figsize = (20, 5)
    modes = threshold_successes.keys()
    draw_succ_with_thresholds(
        modes,
        thresholds,
        threshold_successes,
        threshold_anchors,
        gt_trajs,
        figsize,
        save_dir,
        "Figure7",
    )


def figure8(
    otb_dataset, high_algorithm, high_mcct_algorithm, high_experts, color_map, save_dir
):
    target_seqs = ["Bird1", "Tiger1"]
    draw_result(
        otb_dataset,
        high_mcct_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure8",
        target_seqs,
        show=["frame"],
    )

    draw_result(
        otb_dataset,
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure8",
        target_seqs,
        show=["frame"],
    )


def figure9(
    otb_dataset, high_algorithm, high_hdt_algorithm, high_experts, color_map, save_dir
):
    target_seqs = ["Human4_2", "Skating1"]
    draw_result(
        otb_dataset,
        high_hdt_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure9",
        target_seqs,
        show=["frame"],
    )

    draw_result(
        otb_dataset,
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure9",
        target_seqs,
        show=["frame"],
    )


def figure10(
    datasets,
    datasets_name,
    superfast_algorithm,
    superfast_experts,
    fast_algorithm,
    fast_experts,
    normal_algorithm,
    normal_experts,
    save_dir,
):
    rank_table(
        datasets,
        datasets_name,
        superfast_algorithm,
        superfast_experts,
        save_dir,
        "SuperFast_Offline",
    )
    rank_table(
        datasets, datasets_name, fast_algorithm, fast_experts, save_dir, "Fast_Offline"
    )
    rank_table(
        datasets,
        datasets_name,
        normal_algorithm,
        normal_experts,
        save_dir,
        "Normal_Offline",
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
    metrics=["AUC", "DP", "FPS"],
):
    trackers = experts + baselines + [algorithm]

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
        baselines + [algorithm],
        experts,
        mean_values,
        save_dir,
        filename=filename,
    )


def threshold_table(
    algorithm_name,
    thresholds,
    threshold_successes,
    threshold_precisions,
    threshold_anchor_successes,
    threshold_anchors,
    gt_trajs,
    save_dir,
):
    modes = list(threshold_successes.keys())
    mean_values = {
        "AUC": get_mean_succ(thresholds, modes, threshold_successes).round(3),
        "DP": get_mean_prec(thresholds, modes, threshold_precisions).round(3),
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


def rank_table(
    datasets, datasets_name, algorithm_name, experts_name, save_dir, filename
):
    error_rets = {}
    loss_rets = {}
    for dataset, dataset_name in zip(datasets, datasets_name):
        ope = OPEBenchmark(dataset, dataset_name)

        _error_rets = {}
        _loss_rets = {}
        for expert_name in experts_name:
            error_ret, loss_ret = ope.eval_loss(algorithm_name, expert_name)
            _error_rets[expert_name] = error_ret
            _loss_rets[expert_name] = loss_ret
        error_rets[dataset_name] = _error_rets
        loss_rets[dataset_name] = _loss_rets

    make_rank_table(
        datasets_name, experts_name, error_rets, loss_rets, save_dir, filename
    )


def main(experiments, all_experts, all_experts_name):
    save_dir = Path(path_config.VISUALIZATION_PATH)
    os.makedirs(save_dir, exist_ok=True)

    total_colors = sns.color_palette("hls", len(all_experts) + 1).as_hex()[::-1]
    color_map = {
        tracker_name: color
        for tracker_name, color in zip(all_experts, total_colors[1:])
    }
    color_map["MCCT"] = total_colors[0]
    color_map["HDT"] = total_colors[0]
    color_map["Random"] = total_colors[0]
    color_map["WithoutOffline"] = total_colors[0]
    color_map["WithoutDelay"] = total_colors[0]
    color_map["AAA"] = total_colors[0]

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
        anchor_frame_rets,
    ) = evaluate(datasets, datasets_name, all_experts, [], None)
    figure1(
        datasets_name, all_experts, all_experts_name, success_rets, color_map, save_dir
    )

    # vot = select_datasets("VOT2018")
    # figure2(vot, all_experts, color_map, save_dir)

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

    # Table6
    score_table(
        datasets_name,
        super_fast_algorithm,
        [],
        superfast_experts,
        superfast_anchor_success_rets,
        None,
        None,
        save_dir,
        filename="Table6",
        metrics=["AUC"],
    )

    # # figure4(
    # #     datasets_name,
    # #     high_algorithm,
    # #     high_experts,
    # #     high_successes,
    # #     high_precisions,
    # #     save_dir,
    # # )
    # # otb = select_datasets("OTB")
    # # tpl = select_datasets("TPL")
    # # figure6(otb, tpl, high_algorithm, high_experts, save_dir)
    # # figure8(otb, high_algorithm, high_baselines[1], high_experts, save_dir)
    # # figure9(otb, high_algorithm, high_baselines[0], high_experts, save_dir)

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

    # Figure10
    figure10(
        datasets,
        datasets_name,
        superfast_algorithm,
        superfast_experts,
        fast_algorithm,
        fast_experts,
        normal_algorithm,
        normal_experts,
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

    # Tuning - AAA
    # target_modes = ["SuperFast", "Fast", "Normal"]
    # (
    #     threshold_successes, threshold_precisions, threshold_anchor_successes, threshold_anchors, gt_trajs
    # ) = get_tuning_results(path_config.EVALUATION_PATH, "AAA", target_modes)
    # thresholds = list(threshold_successes[target_modes[0]].keys())
    # threshold_table("AAA", thresholds, threshold_successes, threshold_precisions, threshold_anchor_successes, threshold_anchors, gt_trajs, save_dir)
    # figure7(thresholds, threshold_successes, threshold_anchors, gt_trajs, save_dir)

    # Tuning-HDT
    # target_modes = ["SuperFast", "Fast", "Normal", "SiamDW", "SiamRPN++"]
    # (
    #     threshold_successes, threshold_precisions, threshold_anchor_successes, threshold_anchors, gt_trajs
    # ) = get_tuning_results(path_config.EVALUATION_PATH, "HDT", target_modes)
    # thresholds = list(threshold_successes[target_modes[0]].keys())
    # threshold_table("HDT", thresholds, threshold_successes, threshold_precisions, threshold_anchor_successes, threshold_anchors, gt_trajs, save_dir)


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
        "DaSiamRPN (ECCV 2018)",
        "DiMP (ICCV 2019)",
        "DROL (AAAI 2020)",
        "KYS (ECCV 2020)",
        "Ocean (ECCV 2020)",
        "PrDiMP (CVPR 2020)",
        "RPT (ECCVW 2020)",
        "SiamBAN (CVPR 2020)",
        "SiamCAR (CVPR 2020)",
        "SiamDW (CVPR 2019)",
        "SiamFC++ (AAAI 2020)",
        "SiamRPN (CVPR 2018)",
        "SiamRPN++ (CVPR 2019)",
        "SPM (CVPR 2019)",
    ]

    super_fast_algorithm = "AAA/SuperFast/0.70"
    fast_algorithm = "AAA/Fast/0.72"
    normal_algorithm = "AAA/Normal/0.74"
    siamdw_algorithm = "AAA/SiamDW/0.73"
    siamrpn_algorithm = "AAA/SiamRPN++/0.75"

    super_fast_baselines = [
        "Random/SuperFast",
        "HDT/SuperFast/1.00",
        "MCCT/SuperFast/0.10",
        "WithoutDelay/SuperFast",
    ]
    fast_baselines = [
        "Random/Fast",
        "HDT/Fast/0.96",
        "MCCT/Fast/0.10",
        "WithoutDelay/Fast",
    ]
    normal_baselines = [
        "Random/Normal",
        "HDT/Normal/0.05",
        "MCCT/Normal/0.10",
        "WithoutDelay/Normal",
    ]
    siamdw_baselines = [
        "Random/SiamDW",
        "HDT/SiamDW/0.05",
        "MCCT/SiamDW/0.10",
        "WithoutDelay/SiamDW",
    ]
    siamrpn_baselines = [
        "Random/SiamRPN++",
        "HDT/SiamRPN++/0.05",
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
