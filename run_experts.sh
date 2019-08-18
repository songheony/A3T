trackers=("ATOM" "CSRDCF" "DaSiamRPN" "ECO" "MDNet" "SiamDW" "SiamFC" "SiamRPN" "SiamRPN++" "Staple" "STRCF" "TADT" "Vital")
len=${#trackers[@]}
for (( i=0; i<$len; i++ )); do
    python ./run_expert.py -e ${trackers[$i]} -d NFS
done
