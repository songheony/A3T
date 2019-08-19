algorithms=("Ours" "Max" "Average" "MCCT")
# trackers=("ATOM" "CSRDCF" "DaSiamRPN" "ECO" "MDNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "Staple" "STRCF" "TADT" "Vital")
trackers=("ATOM" "DaSiamRPN" "ECO" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "Staple" "STRCF" "TADT")
len=${#algorithms[@]}
for (( i=0; i<$len; i++ )); do
    python ./run_algorithm.py -a ${algorithms[$i]} -e "${trackers[@]}" -d OTB
done