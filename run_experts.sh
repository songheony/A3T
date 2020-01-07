datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT" "Got10K")

goods=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#goods[@]}; i++ )); do
        python ./track_expert.py -e ${goods[$i]} -d ${datasets[$j]}
    done
done

bads=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#bads[@]}; i++ )); do
        python ./track_expert.py -e ${bads[$i]} -d ${datasets[$j]}
    done
done

siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#siamdws[@]}; i++ )); do
        python ./track_expert.py -e ${siamdws[$i]} -d ${datasets[$j]}
    done
done

siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#siamrpnpps[@]}; i++ )); do
        python ./track_expert.py -e ${siamrpnpps[$i]} -d ${datasets[$j]}
    done
done

thors=("THOR_SiamFC_Dynamic_OTB" "THOR_SiamFC_Dynamic_VOT" "THOR_SiamFC_Ensemble_OTB" "THOR_SiamFC_Ensemble_VOT" "THOR_SiamMask_Dynamic_OTB" "THOR_SiamMask_Dynamic_VOT" "THOR_SiamMask_Ensemble_OTB" "THOR_SiamMask_Ensemble_VOT" "THOR_SiamRPN_Dynamic_OTB" "THOR_SiamRPN_Dynamic_VOT" "THOR_SiamRPN_Ensemble_OTB" "THOR_SiamRPN_Ensemble_VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#thors[@]}; i++ )); do
        python ./track_expert.py -e ${thors[$i]} -d ${datasets[$j]}
    done
done