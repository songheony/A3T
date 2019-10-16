datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
experts=("ATOM" "DaSiamRPN" "DiMP" "ECO" "GradNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
baselines=("Average" "Max" "MCCT")
trackers=("${experts[@]}" "${baselines[@]}")
for (( j=0; j<${#datasets[@]}; j++ )); do
    python ./run_evaluation.py -a AAA_select_0.0_0.77_True_False_False_True_True_True_True -d ${datasets[$j]} -t ${trackers[@]}
done
