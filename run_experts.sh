datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT" "Got10K")

high_experts=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#high_experts[@]}; i++ )); do
        python ./track_expert.py -e ${high_experts[$i]} -d ${datasets[$j]}
    done
done

low_experts=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#low_experts[@]}; i++ )); do
        python ./track_expert.py -e ${low_experts[$i]} -d ${datasets[$j]}
    done
done

siamdw_experts=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#siamdw_experts[@]}; i++ )); do
        python ./track_expert.py -e ${siamdw_experts[$i]} -d ${datasets[$j]}
    done
done

siamrpn_experts=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#siamrpn_experts[@]}; i++ )); do
        python ./track_expert.py -e ${siamrpn_experts[$i]} -d ${datasets[$j]}
    done
done

new_experts=("DiMP" "PrDiMP" "SiamBAN" "SiamRCNN" "TRASFUST")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#high_experts[@]}; i++ )); do
        python ./track_expert.py -e ${high_experts[$i]} -d ${datasets[$j]}
    done
done