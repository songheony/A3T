import os
import pickle
from pathlib import Path
import seaborn as sns
from select_options import select_datasets
from evaluations.eval_trackers import evaluate
from evaluations.ope_benchmark import OPEBenchmark
from visualizes.draw_figures import draw_pie, draw_result, draw_graph, draw_curves, draw_rank, draw_succ_with_thresholds
from visualizes.draw_tables import make_score_table
from path_config import VISUALIZATION_PATH


def get_tuning_results(tune_dir, modes, thresholds):
    dataset_name = "GOT10K"
    dataset = select_datasets(dataset_name)
    ope = OPEBenchmark(dataset)

    threshold_successes = {mode: {} for mode in modes}
    threshold_anchors = {mode: {} for mode in modes}

    for mode in modes:
        for threshold in thresholds:
            algorithm_name = f"AAA_{mode}_{threshold:.2f}"
            success_path = tune_dir / algorithm_name / "success.pkl"
            success = pickle.loads(success_path.read_bytes())

            threshold_successes[mode][threshold] = success
            anchor_frames = ope.get_anchor_frames(algorithm_name)
            threshold_anchors[mode][threshold] = anchor_frames

    return threshold_successes, threshold_anchors


def figure1(datasets_name, all_experts, all_experts_name, all_successes, color_map, save_dir):
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


def figure6(otb_dataset, tpl_dataset, high_algorithm, high_experts, color_map, save_dir):
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


def figure7(threshold_successes, threshold_anchors, save_dir):
    figsize = (20, 5)
    thresholds = sorted(list(threshold_successes.values())[0].keys())
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


def figure8(otb_dataset, high_algorithm, high_mcct_algorithm, high_experts, color_map, save_dir):
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


def figure9(otb_dataset, high_algorithm, high_hdt_algorithm, high_experts, color_map, save_dir):
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


def main(experiments, all_experts, all_experts_name):
    save_dir = Path(VISUALIZATION_PATH)
    os.makedirs(save_dir, exist_ok=True)

    datasets_name = ["OTB2015", "OTB2015-80%", "TColor128", "UAV123", "NFS", "LaSOT", "VOT2018"]
    datasets = [select_datasets(dataset_name) for dataset_name in datasets_name]

    total_colors = sns.color_palette("hls", len(all_experts) + 2).as_hex()[::-1][1:]
    color_map = {tracker_name: color for tracker_name, color in zip(all_experts, total_colors[1:])}
    color_map["MCCT"] = total_colors[0]
    color_map["HDT"] = total_colors[0]
    color_map["Random"] = total_colors[0]
    color_map["Max"] = total_colors[0]
    color_map["AAA"] = total_colors[0]

    # All
    tracking_time_rets, success_rets, precision_rets, norm_precision_rets, anchor_success_rets, anchor_precision_rets, anchor_norm_precision_rets, error_rets, loss_rets, anchor_frame_rets = evaluate(datasets, datasets_name, all_experts, [], None)
    figure1(datasets_name, all_experts, all_experts_name, success_rets, color_map, save_dir)
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
    exit()

    # High
    high_algorithm, high_baselines, high_experts = experiments["High"]
    eval_save = eval_dir / "High" / "eval.pkl"
    (
        high_successes,
        high_precisions,
        high_tracking_times,
        high_anchor_frames,
        high_anchor_successes,
        high_anchor_precisions,
        high_offline_successes,
        high_offline_precisions,
        high_regret_gts,
        high_regret_offlines,
    ) = pickle.loads(eval_save.read_bytes())

    table1(
        datasets_name,
        high_algorithm,
        high_baselines,
        high_experts,
        high_successes,
        high_precisions,
        save_dir,
    )
    table6(
        datasets_name,
        high_algorithm,
        high_experts,
        high_anchor_successes,
        high_anchor_precisions,
        save_dir,
    )

    vot = select_datasets("VOT")
    figure2(vot, high_algorithm, high_experts, save_dir)
    figure4(
        datasets_name,
        high_algorithm,
        high_experts,
        high_successes,
        high_precisions,
        save_dir,
    )
    otb = select_datasets("OTB")
    tpl = select_datasets("TPL")
    figure6(otb, tpl, high_algorithm, high_experts, save_dir)
    figure8(otb, high_algorithm, high_baselines[1], high_experts, save_dir)
    figure9(otb, high_algorithm, high_baselines[0], high_experts, save_dir)

    # Low
    low_algorithm, low_baselines, low_experts = experiments["Low"]
    eval_save = eval_dir / "Low" / "eval.pkl"
    (
        low_successes,
        low_precisions,
        low_tracking_times,
        low_anchor_frames,
        low_anchor_successes,
        low_anchor_precisions,
        low_offline_successes,
        low_offline_precisions,
        low_regret_gts,
        low_regret_offlines,
    ) = pickle.loads(eval_save.read_bytes())
    table2(
        datasets_name,
        low_algorithm,
        low_baselines,
        low_experts,
        low_successes,
        low_precisions,
        save_dir,
    )

    # Mix
    mix_algorithm, mix_baselines, mix_experts = experiments["Mix"]
    eval_save = eval_dir / "Mix" / "eval.pkl"
    (
        mix_successes,
        mix_precisions,
        mix_tracking_times,
        mix_anchor_frames,
        mix_anchor_successes,
        mix_anchor_precisions,
        mix_offline_successes,
        mix_offline_precisions,
        mix_regret_gts,
        mix_regret_offlines,
    ) = pickle.loads(eval_save.read_bytes())
    table3(
        datasets_name,
        mix_algorithm,
        mix_baselines,
        mix_experts,
        mix_successes,
        mix_precisions,
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

    # Tuning
    threshold_successes, threshold_anchors = get_tuning_results(tune_dir)
    figure7(threshold_successes, threshold_anchors, save_dir)


if __name__ == "__main__":
    high_algorithm = "AAA_High_0.00_0.69_False_False_False_True_True_True_True"
    low_algorithm = "AAA_Low_0.00_0.60_False_False_False_True_True_True_True"
    mix_algorithm = "AAA_Mix_0.00_0.65_False_False_False_True_True_True_True"
    siamdw_algorithm = "AAA_SiamDW_0.00_0.67_False_False_False_True_True_True_True"
    siamrpn_algorithm = "AAA_SiamRPN++_0.00_0.61_False_False_False_True_True_True_True"

    high_baselines = ["HDT_High_0.98", "MCCT_High_0.10", "Random_High", "Max_High"]
    low_baselines = ["HDT_Low_0.32", "MCCT_Low_0.10", "Random_Low", "Max_Low"]
    mix_baselines = ["HDT_Mix_0.94", "MCCT_Mix_0.10", "Random_Mix", "Max_Mix"]
    siamdw_baselines = [
        "HDT_SiamDW_0.98",
        "MCCT_SiamDW_0.10",
        "Random_SiamDW",
        "Max_SiamDW",
    ]
    siamrpn_baselines = [
        "HDT_SiamRPN++_0.74",
        "MCCT_SiamRPN++_0.10",
        "Random_SiamRPN++",
        "Max_SiamRPN++",
    ]
    all_experts = [
        "ATOM",
        "DaSiamRPN",
        "DiMP-50",
        "DROL",
        "GradNet",
        "KYS",
        "Ocean",
        "PrDiMP-50",
        "RLS-RTMDNet",
        "SiamBAN",
        "SiamCAR",
        "SiamDW",
        "SiamFC++",
        "SiamMCF",
        "SiamRPN",
        "SiamRPN++",
        "SPM",
        "Staple",
        "THOR",
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
        "RLS-RTMDNet (CVPR 2020)",
        "SiamBAN (CVPR 2020)",
        "SiamCAR (CVPR 2020)",
        "SiamDW (CVPR 2019)",
        "SiamFC++ (AAAI 2020)",
        "SiamMCF (ECCVW 2018)",
        "SiamRPN (CVPR 2018)",
        "SiamRPN++ (CVPR 2019)",
        "SPM (CVPR 2019)",
        "Staple (CVPR 2016)",
        "THOR (BMVC 2019)",
    ]
    high_experts = ["ATOM", "DaSiamRPN", "SiamMCF", "SiamRPN++", "SPM", "THOR"]
    low_experts = ["GradNet", "MemTrack", "SiamDW", "SiamFC", "SiamRPN", "Staple"]
    mix_experts = ["ATOM", "SiamRPN++", "SPM", "MemTrack", "SiamFC", "Staple"]
    siamdw_experts = [
        "SiamDW_SiamFCRes22",
        "SiamDW_SiamFCIncep22",
        "SiamDW_SiamFCNext22",
        "SiamDW_SiamRPNRes22",
        "SiamDW_SiamFCRes22_VOT",
        "SiamDW_SiamFCIncep22_VOT",
        "SiamDW_SiamFCNext22_VOT",
        "SiamDW_SiamRPNRes22_VOT",
    ]
    siamrpn_experts = [
        "SiamRPN++_AlexNet",
        "SiamRPN++_AlexNet_OTB",
        "SiamRPN++_ResNet-50",
        "SiamRPN++_ResNet-50_OTB",
        "SiamRPN++_ResNet-50_LT",
        "SiamRPN++_MobileNetV2",
        "SiamRPN++_SiamMask",
    ]

    experiments = {
        "High": (high_algorithm, high_baselines, high_experts),
        "Low": (low_algorithm, low_baselines, low_experts),
        "Mix": (mix_algorithm, mix_baselines, mix_experts),
        "SiamDW": (siamdw_algorithm, siamdw_baselines, siamdw_experts),
        "SiamRPN++": (siamrpn_algorithm, siamrpn_baselines, siamrpn_experts),
    }

    main(experiments, all_experts, all_experts_name)
