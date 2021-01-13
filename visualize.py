import os
import pickle
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
    make_score_table,
)
import path_config


def get_tuning_results(eval_dir, modes):
    dataset_name = "Got10K"
    dataset = select_datasets(dataset_name)
    ope = OPEBenchmark(dataset, dataset_name)

    thresholds = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]
    feature_factor = 11

    threshold_successes = {mode: {} for mode in modes}
    threshold_anchor_successes = {mode: {} for mode in modes}
    threshold_anchors = {mode: {} for mode in modes}

    for mode in modes:
        for threshold in thresholds:
            parameter_name = f"{threshold:.2f}/{feature_factor}"

            if threshold == 0:
                algorithm_name = f"WithoutDelay/{mode}/{feature_factor}"
            else:
                algorithm_name = f"AAA/{mode}/{parameter_name}"

            success_path = (
                Path(eval_dir) / algorithm_name / dataset_name / "success.pkl"
            )
            success = pickle.loads(success_path.read_bytes())

            anchor_success_path = (
                Path(eval_dir) / algorithm_name / dataset_name / "anchor_success.pkl"
            )
            anchor_success = pickle.loads(anchor_success_path.read_bytes())

            anchor_frames = ope.get_anchor_frames(algorithm_name)
            threshold_successes[mode][parameter_name] = success
            threshold_anchor_successes[mode][parameter_name] = anchor_success
            threshold_anchors[mode][parameter_name] = anchor_frames

    gt_trajs = ope.get_gt_trajs()

    return threshold_successes, threshold_anchor_successes, threshold_anchors, gt_trajs


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


def figure2(vot_dataset, high_algorithm, high_experts, color_map, save_dir):
    target_seqs = ["fish1", "nature"]
    draw_result(
        vot_dataset,
        high_algorithm,
        high_experts,
        color_map,
        save_dir / "Figure2",
        target_seqs,
        show=["frame"],
        gt=True,
        best=True,
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
):
    trackers = experts + baselines + [algorithm]
    mean_values = {
        "AUC": get_mean_succ(trackers, datasets_name, successes),
        "DP": get_mean_prec(trackers, datasets_name, precisions),
        "FPS": get_mean_fps(trackers, datasets_name, tracking_times),
    }
    make_score_table(
        datasets_name,
        baselines + [algorithm],
        experts,
        mean_values,
        save_dir,
        filename=filename,
    )


def main(experiments, all_experts, all_experts_name):
    save_dir = Path(path_config.VISUALIZATION_PATH)
    os.makedirs(save_dir, exist_ok=True)

    datasets_name = [
        "OTB2015",
        "TColor128",
        "UAV123",
        "NFS",
        "LaSOT",
        "VOT2018",
    ]
    datasets = [select_datasets(dataset_name) for dataset_name in datasets_name]

    total_colors = sns.color_palette("hls", len(all_experts) + 2).as_hex()[::-1][1:]
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

    # All
    # (
    #     tracking_time_rets,
    #     success_rets,
    #     precision_rets,
    #     norm_precision_rets,
    #     anchor_success_rets,
    #     anchor_precision_rets,
    #     anchor_norm_precision_rets,
    #     error_rets,
    #     loss_rets,
    #     anchor_frame_rets,
    # ) = evaluate(datasets, datasets_name, all_experts, [], None)
    # figure1(
    #     datasets_name, all_experts, all_experts_name, success_rets, color_map, save_dir
    # )

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
        superfast_anchor_precision_rets,
        superfast_tracking_time_rets,
        save_dir,
        filename="Table6",
    )
    exit()

    # vot = select_datasets("VOT")
    # figure2(vot, high_algorithm, high_experts, save_dir)
    # figure4(
    #     datasets_name,
    #     high_algorithm,
    #     high_experts,
    #     high_successes,
    #     high_precisions,
    #     save_dir,
    # )
    # otb = select_datasets("OTB")
    # tpl = select_datasets("TPL")
    # figure6(otb, tpl, high_algorithm, high_experts, save_dir)
    # figure8(otb, high_algorithm, high_baselines[1], high_experts, save_dir)
    # figure9(otb, high_algorithm, high_baselines[0], high_experts, save_dir)

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

    exit()

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

    # Tuning
    target_modes = ["SuperFast", "Fast", "Normal"]
    (
        threshold_successes,
        threshold_anchor_successes,
        threshold_anchors,
        gt_trajs,
    ) = get_tuning_results(path_config.EVALUATION_PATH, target_modes)
    thresholds = sorted(list(threshold_successes[target_modes[0]].keys()))
    figure7(thresholds, threshold_successes, threshold_anchors, gt_trajs, save_dir)


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

    super_fast_algorithm = "AAA/SuperFast/0.88/11"
    fast_algorithm = "AAA/Fast/0.92/11"
    normal_algorithm = "AAA/Normal/0.87/11"
    siamdw_algorithm = "AAA/SiamDW/0.67"
    siamrpn_algorithm = "AAA/SiamRPN++/0.61"

    super_fast_baselines = [
        # "Random/SuperFast",
        # "WithoutOffline/SuperFast",
        # "WithoutDelay/SuperFast",
    ]
    fast_baselines = []
    normal_baselines = []
    siamdw_baselines = []
    siamrpn_baselines = []
    # super_fast_baselines = [
    #     "HDT/SuperFast/0.98",
    #     "MCCT/SuperFast/0.10",
    #     "Random/SuperFast",
    #     "Max/SuperFast",
    #     "Without delay/SuperFast",
    # ]
    # fast_baselines = [
    #     "HDT/Fast/0.98",
    #     "MCCT/Fast/0.10",
    #     "Random/Fast",
    #     "Max/Fast",
    #     "Without delay/Fast",
    # ]
    # normal_baselines = [
    #     "HDT/Normal/0.98",
    #     "MCCT/Normal/0.10",
    #     "Random/Normal",
    #     "Max/Normal",
    #     "Without delay/Normal",
    # ]
    # siamdw_baselines = [
    #     "HDT/SiamDW/0.98",
    #     "MCCT/SiamDW/0.10",
    #     "Random/SiamDW",
    #     "Max/SiamDW",
    #     "Without delay/SiamDW",
    # ]
    # siamrpn_baselines = [
    #     "HDT/SiamRPN++/0.98",
    #     "MCCT/SiamRPN++/0.10",
    #     "Random/SiamRPN++",
    #     "Max/SiamRPN++",
    #     "Without delay/SiamRPN++",
    # ]

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
