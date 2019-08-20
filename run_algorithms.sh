algorithms=("Average" "Max" "MCCT" "AAA_similar" "AAA_overlap")
experts=("ATOM" "DaSiamRPN" "ECO" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "Staple" "STRCF" "TADT")
len=${#algorithms[@]}
for (( i=0; i<$len; i++ )); do
    python ./run_algorithm.py -a ${algorithms[$i]} -e "${experts[@]}" -d OTB
done