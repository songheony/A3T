experts=("ATOM" "DaSiamRPN" "DiMP" "ECO" "GradNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
for (( j=0; j<${#datasets[@]}; j++ )); do
    python ./run_tuning.py -a AAA_select -e $experts -d ${datasets[$j]}
done
