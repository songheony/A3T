high_experts=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
python ./track_tuning.py -e ${high_experts[@]} -n High
python ./track_tuning.py -a HDT -e ${high_experts[@]} -n High

low_experts=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
python ./track_tuning.py -e ${low_experts[@]} -n Low
python ./track_tuning.py -a HDT -e ${low_experts[@]} -n Low

mix_experts=("ATOM" "SiamRPN++" "SPM" "MemTrack" "SiamFC" "Staple")
python ./track_tuning.py -e ${mix_experts[@]} -n Mix
python ./track_tuning.py -a HDT -e ${mix_experts[@]} -n Mix

siamdw_experts=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
python ./track_tuning.py -e ${siamdw_experts[@]} -n SiamDW
python ./track_tuning.py -a HDT -e ${siamdw_experts[@]} -n SiamDW

siamrpn_experts=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
python ./track_tuning.py -e ${siamrpn_experts[@]} -n SiamRPN++
python ./track_tuning.py -a HDT -e ${siamrpn_experts[@]} -n SiamRPN++