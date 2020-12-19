super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
python ./track_tuning.py -e ${super_fast_experts[@]} -n SuperFast
python ./track_tuning.py -a HDT -e ${super_fast_experts[@]} -n SuperFast

fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
python ./track_tuning.py -e ${fast_experts[@]} -n Fast
python ./track_tuning.py -a HDT -e ${fast_experts[@]} -n Fast

normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "SiamMCF")
python ./track_tuning.py -e ${normal_experts[@]} -n Normal
python ./track_tuning.py -a HDT -e ${normal_experts[@]} -n Normal

# siamdw_experts=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
# python ./track_tuning.py -e ${siamdw_experts[@]} -n SiamDW
# python ./track_tuning.py -a HDT -e ${siamdw_experts[@]} -n SiamDW

# siamrpn_experts=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
# python ./track_tuning.py -e ${siamrpn_experts[@]} -n SiamRPN++
# python ./track_tuning.py -a HDT -e ${siamrpn_experts[@]} -n SiamRPN++