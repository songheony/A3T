trackers=("ATOM" "DaSiamRPN" "ECO" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "Staple" "STRCF" "TADT" "Average" "Max" "MCCT" "AAA_similar" "AAA_overlap")
python ./run_evaluation.py -t "${trackers[@]}" -d OTB
