datasets=("OTB")
experts=("ATOM" "DaSiamRPN" "ECO" "GradNet" "MemDTC" "MemTrack" "SiamDW" "SiamMCF" "SiamRPN++" "SPM" "THOR")
baselines=("Average" "Max" "MCCT")
trackers=("${experts[@]}" "${baselines[@]}")
for (( j=0; j<${#datasets[@]}; j++ )); do
    python ./run_evaluation.py -a AAA_select -d ${datasets[$j]} -t ${trackers[@]}
done
