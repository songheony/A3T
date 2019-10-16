datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT" "Got10K")
# "ATOM" "DaSiamRPN" "ECO" "GradNet" "MemDTC" "MemTrack" "SiamDW" "SiamMCF" "SiamRPN++" "SPM" "THOR"
experts=("ATOM" "DaSiamRPN" "ECO" "GradNet" "MemDTC" "MemTrack" "SiamDW" "SiamMCF" "SiamRPN++" "SPM" "THOR")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#experts[@]}; i++ )); do
        python ./run_expert.py -e ${experts[$i]} -d ${datasets[$j]}
    done
done
