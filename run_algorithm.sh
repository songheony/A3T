datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
experts=("ATOM" "DaSiamRPN" "DiMP" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
thors=("THOR_SiamFC_Dynamic_OTB" "THOR_SiamFC_Dynamic_VOT" "THOR_SiamFC_Ensemble_OTB" "THOR_SiamFC_Ensemble_VOT" "THOR_SiamMask_Dynamic_OTB" "THOR_SiamMask_Dynamic_VOT" "THOR_SiamMask_Ensemble_OTB" "THOR_SiamMask_Ensemble_VOT" "THOR_SiamRPN_Dynamic_OTB" "THOR_SiamRPN_Dynamic_VOT" "THOR_SiamRPN_Ensemble_OTB" "THOR_SiamRPN_Ensemble_VOT")
thresholds=(0.76 0.77 0.78)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -e ${siamdws[@]} -s -r ${thresholds[$i]}
    done
done
