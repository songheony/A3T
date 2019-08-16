trackers=("ATOM" "BACF" "CSRDCF" "DaSiamRPN" "ECO" "MDNet" "SAMF" "SiamDW" "SiamFC" "SiamRPN" "Staple" "STRCF" "TADT" "Vital")
len=${#trackers[@]}
for (( i=0; i<$len; i++ )); do
    python ./run_expert.py -e ${trackers[$i]} -d OTB
done