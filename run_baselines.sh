datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
experts=("ATOM" "DaSiamRPN" "DiMP" "ECO" "GradNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
baselines=("Average" "Max" "MCCT")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./run_baseline.py -a ${baselines[$i]} -d ${datasets[$j]} -e ${experts[@]}
    done
done
