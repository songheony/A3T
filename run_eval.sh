algorithms=("AAA_SiamDW_0.00_0.67_False_False_False_True_True_True_True" "AAA_SiamDW_0.00_0.72_False_False_False_True_True_True_True")
baselinedw=("Random_SiamDW" "MCCT_SiamDW" "Max_SiamDW")
baselinerpn=("Random_SiamRPN++" "MCCT_SiamRPN++" "Max_SiamRPN++")
experts=("ATOM" "DaSiamRPN" "DiMP" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
thors=("THOR_SiamFC_Dynamic_OTB" "THOR_SiamFC_Dynamic_VOT" "THOR_SiamFC_Ensemble_OTB" "THOR_SiamFC_Ensemble_VOT" "THOR_SiamMask_Dynamic_OTB" "THOR_SiamMask_Dynamic_VOT" "THOR_SiamMask_Ensemble_OTB" "THOR_SiamMask_Ensemble_VOT" "THOR_SiamRPN_Dynamic_OTB" "THOR_SiamRPN_Dynamic_VOT" "THOR_SiamRPN_Ensemble_OTB" "THOR_SiamRPN_Ensemble_VOT")
python ./visualize_eval.py -a ${algorithms[@]} -e ${siamdws[@]} -d SiamDW -b ${baselinedw[@]}
