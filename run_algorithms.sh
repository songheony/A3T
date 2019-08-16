algorithms=("Ours" "Max" "Average" "MCCT")
trackers=("ATOM" "BACF" "CSRDCF" "DaSiamRPN" "ECO" "ECO_new" "MDNet" "SAMF" "SiamDW" "SiamFC" "SiamRPN" "Staple" "STRCF" "TADT" "Vital")
len=${#algorithms[@]}
for (( i=0; i<$len; i++ )); do
    python ./run_algorithm.py -a ${algorithms[$i]} -e "${trackers[@]}" -d OTB
done