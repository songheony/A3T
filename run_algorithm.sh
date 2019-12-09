datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
experts=("ATOM" "DaSiamRPN" "DiMP" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
# 0.67 0.72
# 0.67 0.72 -z
siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
# 0.61 0.73
# 0.66 0.76 -z
siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
thors=("THOR_SiamFC_Dynamic_OTB" "THOR_SiamFC_Dynamic_VOT" "THOR_SiamFC_Ensemble_OTB" "THOR_SiamFC_Ensemble_VOT" "THOR_SiamMask_Dynamic_OTB" "THOR_SiamMask_Dynamic_VOT" "THOR_SiamMask_Ensemble_OTB" "THOR_SiamMask_Ensemble_VOT" "THOR_SiamRPN_Dynamic_OTB" "THOR_SiamRPN_Dynamic_VOT" "THOR_SiamRPN_Ensemble_OTB" "THOR_SiamRPN_Ensemble_VOT")
thresholds=(0.66 0.76)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n SiamRPN++ -e ${siamrpnpps[@]} -r ${thresholds[$i]} -z
    done
done
