# all=("ATOM" "DaSiamRPN" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
# python ./track_tuning.py -e ${all[@]} -n All

goods=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
python ./track_tuning.py -e ${goods[@]} -n Good

bads=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
python ./track_tuning.py -e ${bads[@]} -n Bad

mixs=("ATOM" "SiamRPN++" "SPM" "MemTrack" "SiamFC" "Staple")
python ./track_tuning.py -e ${mixs[@]} -n Mix

siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
python ./track_tuning.py -e ${siamdws[@]} -n SiamDW

siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
python ./track_tuning.py -e ${siamrpnpps[@]} -n SiamRPN++