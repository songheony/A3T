experts=("ATOM" "DaSiamRPN" "ECO" "SiamDW" "SiamMCF" "SiamRPN++" "SPM" "THOR")
python ./run_tuning.py -a AAA_select -e ${experts[@]}
