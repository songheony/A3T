datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
experts=("ATOM" "DaSiamRPN" "ECO" "SiamDW" "SiamMCF" "SiamRPN++" "SPM" "THOR")
thresholds=(0.67)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./run_algorithm.py -a AAA_select -d ${datasets[$j]} -e ${experts[@]} -r ${thresholds[$i]}
    done
done
