experts=("ATOM" "DaSiamRPN" "ECO" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "Staple" "STRCF" "TADT")
len=${#experts[@]}
for (( i=0; i<$len; i++ )); do
    python ./run_expert.py -e ${experts[$i]} -d LaSOT
done
