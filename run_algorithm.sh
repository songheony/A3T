datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
experts=("ATOM" "DaSiamRPN" "DiMP" "ECO" "GradNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
thresholds=(0.76 0.77)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./run_algorithm.py -a AAA_select -d ${datasets[$j]} -e $experts -r ${thresholds[$i]}
    done
done
