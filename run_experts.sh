datasets=("OTB2015" "OTB2015-80%" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT" "Got10K")

ultra_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM" "THOR")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#ultra_fast_experts[@]}; i++ )); do
        python ./track_expert.py -e ${ultra_fast_experts[$i]} -d ${datasets[$j]}
    done
done

fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++" "Staple")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#fast_experts[@]}; i++ )); do
        python ./track_expert.py -e ${fast_experts[$i]} -d ${datasets[$j]}
    done
done

normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "RLS-RTMDNet" "SiamMCF")
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( i=0; i<${#normal_experts[@]}; i++ )); do
        python ./track_expert.py -e ${normal_experts[$i]} -d ${datasets[$j]}
    done
done

# slow_experts=("SiamFC" "SiamR-CNN")
# for (( j=0; j<${#datasets[@]}; j++ )); do
#     for (( i=0; i<${#slow_experts[@]}; i++ )); do
#         python ./track_expert.py -e ${slow_experts[$i]} -d ${datasets[$j]}
#     done
# done

# siamdw_experts=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
# for (( j=0; j<${#datasets[@]}; j++ )); do
#     for (( i=0; i<${#siamdw_experts[@]}; i++ )); do
#         python ./track_expert.py -e ${siamdw_experts[$i]} -d ${datasets[$j]}
#     done
# done

# siamrpn_experts=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
# for (( j=0; j<${#datasets[@]}; j++ )); do
#     for (( i=0; i<${#siamrpn_experts[@]}; i++ )); do
#         python ./track_expert.py -e ${siamrpn_experts[$i]} -d ${datasets[$j]}
#     done
# done
