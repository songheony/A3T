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
from visualizes.draw_tables import make_score_table
import path_config


def get_tuning_results(eval_dir, modes, thresholds):
    dataset_name = "GOT10K"
    dataset = select_datasets(dataset_name)
    ope = OPEBenchmark(dataset)

    threshold_successes = {mode: {} for mode in modes}
    threshold_anchors = {mode: {} for mode in modes}

    for mode in modes:
        for threshold in thresholds:
            algorithm_name = f"AAA/{mode}/{threshold:.2f}"
            success_path = eval_dir / algorithm_name / "success.pkl"
            success = pickle.loads(success_path.read_bytes())

            threshold_successes[mode][threshold] = success
            anchor_frames = ope.get_anchor_frames(algorithm_name)
            threshold_anchors[mode][threshold] = anchor_frames

    return threshold_successes, threshold_anchors


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
    high_algorithm,
    high_baselines,
    high_experts,
    high_successes,
    high_precisions,
    low_algorithm,
    low_baselines,
    low_experts,
    low_successes,
    low_precisions,
    mix_algorithm,
    mix_baselines,
    mix_experts,
    mix_successes,
    mix_precisions,
    color_map,
    save_dir,
):
    figsize = (20, 7)
    high_trackers = high_experts + [high_algorithm]
    low_trackers = low_experts + [low_algorithm]
    mix_trackers = mix_experts + [mix_algorithm]
    draw_rank(
        datasets_name,
        ["High", "Low", "Mix"],
        [high_trackers, low_trackers, mix_trackers],
        [high_successes, low_successes, mix_successes],
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


def figure7(thresholds, threshold_successes, threshold_anchors, save_dir):
    figsize = (20, 5)
    modes = threshold_successes.keys()
    draw_succ_with_thresholds(
        modes,
        thresholds,
        threshold_successes,
        threshold_anchors,
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


def table1(
    datasets_name,
    high_algorithm,
    high_baselines,
    high_experts,
    high_successes,
    high_precisions,
    save_dir,
):
    eval_trackers = high_experts + high_baselines + [high_algorithm]
    make_score_table(
        datasets_name,
        eval_trackers,
        len(high_experts),
        high_successes,
        high_precisions,
        save_dir,
        "Table1",
        isvot=True,
    )


def table2(
    datasets_name,
    low_algorithm,
    low_baselines,
    low_experts,
    low_successes,
    low_precisions,
    save_dir,
):
    eval_trackers = low_experts + low_baselines + [low_algorithm]
    make_score_table(
        datasets_name,
        eval_trackers,
        len(low_experts),
        low_successes,
        low_precisions,
        save_dir,
        "Table2",
        isvot=True,
    )


def table3(
    datasets_name,
    mix_algorithm,
    mix_baselines,
    mix_experts,
    mix_successes,
    mix_precisions,
    save_dir,
):
    eval_trackers = mix_experts + mix_baselines + [mix_algorithm]
    make_score_table(
        datasets_name,
        eval_trackers,
        len(mix_experts),
        mix_successes,
        mix_precisions,
        save_dir,
        "Table3",
        isvot=True,
    )


def table4(
    datasets_name,
    siamdw_algorithm,
    siamdw_baselines,
    siamdw_experts,
    siamdw_successes,
    siamdw_precisions,
    save_dir,
):
    siamdw_algorithms = siamdw_baselines + [siamdw_algorithm]
    make_score_table(
        datasets_name,
        siamdw_algorithms,
        siamdw_experts,
        siamdw_successes,
        siamdw_precisions,
        save_dir,
        "Table4",
        isvot=True,
    )


def table5(
    datasets_name,
    siamrpn_algorithm,
    siamrpn_baselines,
    siamrpn_experts,
    siamrpn_successes,
    siamrpn_precisions,
    save_dir,
):
    siamrpn_algorithms = siamrpn_baselines + [siamrpn_algorithm]
    make_score_table(
        datasets_name,
        siamrpn_algorithms,
        siamrpn_algorithms,
        siamrpn_successes,
        siamrpn_precisions,
        save_dir,
        "Table5",
        isvot=True,
    )


def table6(
    datasets_name,
    high_algorithm,
    high_experts,
    high_anchor_successes,
    high_anchor_precisions,
    save_dir,
):
    make_score_table(
        datasets_name,
        [high_algorithm],
        high_experts,
        high_anchor_successes,
        high_anchor_precisions,
        save_dir,
        "Table6",
        isvot=True,
    )


def main(experiments, all_experts, all_experts_name, thresholds):
    save_dir = Path(path_config.VISUALIZATION_PATH)
    os.makedirs(save_dir, exist_ok=True)

    datasets_name = [
        "OTB2015",
        "OTB2015-80%",
        "OTB2015-60%",
        "OTB2015-40%",
        "OTB2015-20%",
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
    color_map["Max"] = total_colors[0]
    color_map["Without delay"] = total_colors[0]
    color_map["AAA"] = total_colors[0]

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
    make_score_table(
        datasets_name,
        [],
        all_experts,
        success_rets,
        precision_rets,
        save_dir,
        "Table1",
        drop_dp=True,
    )

    # Tuning
    threshold_successes, threshold_anchors = get_tuning_results(
        path_config.EVALUATION_PATH, experiments.keys(), thresholds
    )
    figure7(thresholds, threshold_successes, threshold_anchors, save_dir)
    exit()

    # Super Fast
    super_fast_algorithm, super_fast_baselines, super_fast_experts = experiments[
        "SuperFast"
    ]
    (
        super_fast_tracking_time_rets,
        super_fast_success_rets,
        super_fast_precision_rets,
        super_fast_norm_precision_rets,
        super_fast_anchor_success_rets,
        super_fast_anchor_precision_rets,
        super_fast_anchor_norm_precision_rets,
        super_fast_error_rets,
        super_fast_loss_rets,
        super_fast_anchor_frame_rets,
    ) = evaluate(
        datasets,
        datasets_name,
        super_fast_experts,
        super_fast_baselines,
        super_fast_algorithm,
    )

    table1(
        datasets_name,
        super_fast_algorithm,
        super_fast_baselines,
        super_fast_experts,
        super_fast_success_rets,
        super_fast_precision_rets,
        save_dir,
    )
    table6(
        datasets_name,
        super_fast_algorithm,
        super_fast_experts,
        super_fast_anchor_success_rets,
        super_fast_anchor_precision_rets,
        save_dir,
    )

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

    table2(
        datasets_name,
        fast_algorithm,
        fast_baselines,
        fast_experts,
        fast_success_rets,
        fast_precision_rets,
        save_dir,
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

    table3(
        datasets_name,
        normal_algorithm,
        normal_baselines,
        normal_experts,
        normal_success_rets,
        normal_precision_rets,
        save_dir,
    )

    # All
    figure5(
        datasets_name,
        high_algorithm,
        high_baselines,
        high_experts,
        high_successes,
        high_precisions,
        low_algorithm,
        low_baselines,
        low_experts,
        low_successes,
        low_precisions,
        mix_algorithm,
        mix_baselines,
        mix_experts,
        mix_successes,
        mix_precisions,
        save_dir,
    )

    # SiamDW
    siamdw_algorithm, siamdw_baselines, siamdw_experts = experiments["SiamDW"]
    eval_save = eval_dir / "SiamDW" / "eval.pkl"
    (
        siamdw_successes,
        siamdw_precisions,
        siamdw_tracking_times,
        siamdw_anchor_frames,
        siamdw_anchor_successes,
        siamdw_anchor_precisions,
        siamdw_offline_successes,
        siamdw_offline_precisions,
        siamdw_regret_gts,
        siamdw_regret_offlines,
    ) = pickle.loads(eval_save.read_bytes())
    table4(
        datasets_name,
        siamdw_algorithm,
        siamdw_baselines,
        siamdw_experts,
        siamdw_successes,
        siamdw_precisions,
        save_dir,
    )

    # SiamRPN++
    siamrpn_algorithm, siamrpn_baselines, siamrpn_experts = experiments["SiamRPN++"]
    eval_save = eval_dir / "SiamRPN++" / "eval.pkl"
    (
        siamrpn_successes,
        siamrpn_precisions,
        siamrpn_tracking_times,
        siamrpn_anchor_frames,
        siamrpn_anchor_successes,
        siamrpn_anchor_precisions,
        siamrpn_offline_successes,
        siamrpn_offline_precisions,
        siamrpn_regret_gts,
        siamrpn_regret_offlines,
    ) = pickle.loads(eval_save.read_bytes())
    table5(
        datasets_name,
        siamrpn_algorithm,
        siamrpn_baselines,
        siamrpn_experts,
        siamrpn_successes,
        siamrpn_precisions,
        save_dir,
    )


if __name__ == "__main__":
    all_experts = [
        "ATOM",
        "DaSiamRPN",
        "DiMP-50",
        "DROL",
        "GradNet",
        "KYS",
        "Ocean",
        "PrDiMP-50",
        "SiamBAN",
        "SiamCAR",
        "SiamDW",
        "SiamFC++",
        "SiamMCF",
        "SiamRPN",
        "SiamRPN++",
        "SPM",
    ]
    all_experts_name = [
        "ATOM (CVPR 2019)",
        "DaSiamRPN (ECCV 2018)",
        "DiMP (ICCV 2019)",
        "DROL (AAAI 2020)",
        "GradNet (ICCV 2019)",
        "KYS (ECCV 2020)",
        "Ocean (ECCV 2020)",
        "PrDiMP (CVPR 2020)",
        "SiamBAN (CVPR 2020)",
        "SiamCAR (CVPR 2020)",
        "SiamDW (CVPR 2019)",
        "SiamFC++ (AAAI 2020)",
        "SiamMCF (ECCVW 2018)",
        "SiamRPN (CVPR 2018)",
        "SiamRPN++ (CVPR 2019)",
        "SPM (CVPR 2019)",
    ]

    super_fast_algorithm = "AAA/SuperFast/0.69"
    fast_algorithm = "AAA/Fast/0.60"
    normal_algorithm = "AAA/Normal/0.65"
    siamdw_algorithm = "AAA/SiamDW/0.67"
    siamrpn_algorithm = "AAA/SiamRPN++/0.61"

    super_fast_baselines = [
        "HDT/SuperFast/0.98",
        "MCCT/SuperFast/0.10",
        "Random/SuperFast",
        "Max/SuperFast",
        "Without delay/SuperFast",
    ]
    fast_baselines = [
        "HDT/Fast/0.98",
        "MCCT/Fast/0.10",
        "Random/Fast",
        "Max/Fast",
        "Without delay/Fast",
    ]
    normal_baselines = [
        "HDT/Normal/0.98",
        "MCCT/Normal/0.10",
        "Random/Normal",
        "Max/Normal",
        "Without delay/Normal",
    ]
    siamdw_baselines = [
        "HDT/SiamDW/0.98",
        "MCCT/SiamDW/0.10",
        "Random/SiamDW",
        "Max/SiamDW",
        "Without delay/SiamDW",
    ]
    siamrpn_baselines = [
        "HDT/SiamRPN++/0.98",
        "MCCT/SiamRPN++/0.10",
        "Random/SiamRPN++",
        "Max/SiamRPN++",
        "Without delay/SiamRPN++",
    ]

    super_fast_experts = ["DaSiamRPN", "SiamDW", "SiamRPN", "SPM"]
    fast_experts = ["GradNet", "Ocean", "SiamBAN", "SiamCAR", "SiamFC++", "SiamRPN++"]
    normal_experts = ["ATOM", "DiMP", "DROL", "KYS", "PrDiMP", "SiamMCF"]
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

    algorithm_thresholds = np.arange(0.5, 1.0, 0.01)

    main(experiments, all_experts, all_experts_name, algorithm_thresholds)
