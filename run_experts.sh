datasets=("OTB2015" "OTB2015-80%" "OTB2015-60%" "OTB2015-40%" "OTB2015-20%" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT" "Got10K")

# super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
# for (( j=0; j<${#datasets[@]}; j++ )); do
#     for (( i=0; i<${#super_fast_experts[@]}; i++ )); do
#         python ./track_expert.py -e ${super_fast_experts[$i]} -d ${datasets[$j]}
#     done
# done

# fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
# for (( j=0; j<${#datasets[@]}; j++ )); do
#     for (( i=0; i<${#fast_experts[@]}; i++ )); do
#         python ./track_expert.py -e ${fast_experts[$i]} -d ${datasets[$j]}
#     done
# done

# normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "SiamMCF")
# for (( j=0; j<${#datasets[@]}; j++ )); do
#     for (( i=0; i<${#normal_experts[@]}; i++ )); do
#         python ./track_expert.py -e ${normal_experts[$i]} -d ${datasets[$j]}
#     done
# done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#siamdw_experts[@]}; i++ )); do
        python ./track_expert.py -e ${siamdw_experts[$i]} -d ${datasets[$j]}
    done
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#siamrpn_experts[@]}; i++ )); do
        python ./track_expert.py -e ${siamrpn_experts[$i]} -d ${datasets[$j]}
    done
done
