super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
python ./track_tuning.py -e ${super_fast_experts[@]} -n SuperFast
python ./track_tuning.py -a HDT -e ${super_fast_experts[@]} -n SuperFast

fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
python ./track_tuning.py -e ${fast_experts[@]} -n Fast
python ./track_tuning.py -a HDT -e ${fast_experts[@]} -n Fast

normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "SiamMCF")
python ./track_tuning.py -e ${normal_experts[@]} -n Normal
python ./track_tuning.py -a HDT -e ${normal_experts[@]} -n Normal

# siamdw_experts=("SiamDW/SiamFCRes22/OTB" "SiamDW/SiamFCIncep22/OTB" "SiamDW/SiamFCNext22/OTB" "SiamDW/SiamRPNRes22/OTB" "SiamDW/SiamFCRes22/VOT" "SiamDW/SiamFCIncep22/VOT" "SiamDW/SiamFCNext22/VOT" "SiamDW/SiamRPNRes22/VOT")
# python ./track_tuning.py -e ${siamdw_experts[@]} -n SiamDW
# python ./track_tuning.py -a HDT -e ${siamdw_experts[@]} -n SiamDW

# siamrpn_experts=("SiamRPN++/AlexNet/VOT" "SiamRPN++/AlexNet/OTB" "SiamRPN++/ResNet-50/VOT" "SiamRPN++/ResNet-50/OTB" "SiamRPN++/ResNet-50/VOTLT" "SiamRPN++/MobileNetV2/VOT" "SiamRPN++/SiamMask/VOT")
# python ./track_tuning.py -e ${siamrpn_experts[@]} -n SiamRPN++
# python ./track_tuning.py -a HDT -e ${siamrpn_experts[@]} -n SiamRPN++