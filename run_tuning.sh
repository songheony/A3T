super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
python ./track_tuning.py -e ${super_fast_experts[@]} -n SuperFast
python ./track_tuning.py -a HDT -e ${super_fast_experts[@]} -n SuperFast

fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
python ./track_tuning.py -e ${fast_experts[@]} -n Fast
python ./track_tuning.py -a HDT -e ${fast_experts[@]} -n Fast

normal_experts=("ATOM" "DiMP-50" "DROL" "KYS" "PrDiMP-50" "SiamMCF")
python ./track_tuning.py -e ${normal_experts[@]} -n Normal
python ./track_tuning.py -a HDT -e ${normal_experts[@]} -n Normal

# siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
# python ./track_tuning.py -e ${siamdw_experts[@]} -n SiamDW
# python ./track_tuning.py -a HDT -e ${siamdw_experts[@]} -n SiamDW

# siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
# python ./track_tuning.py -e ${siamrpn_experts[@]} -n SiamRPN++
# python ./track_tuning.py -a HDT -e ${siamrpn_experts[@]} -n SiamRPN++