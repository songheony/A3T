experts=("ATOM" "DaSiamRPN" "DiMP" "ECO" "GradNet" "MemDTC" "MemTrack" "RT-MDNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT" "Got10K")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#experts[@]}; i++ )); do
        python ./run_expert.py -e ${experts[$i]} -d ${datasets[$j]}
    done
done
