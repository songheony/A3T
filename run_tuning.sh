super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
python ./track_tuning.py -e ${super_fast_experts[@]} -m SuperFast
python ./track_tuning.py -a HDT -e ${super_fast_experts[@]} -m SuperFast
python ./track_tuning.py -a MCCT -e ${super_fast_experts[@]} -m SuperFast

fast_experts=("Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
python ./track_tuning.py -e ${fast_experts[@]} -m Fast
python ./track_tuning.py -a HDT -e ${fast_experts[@]} -m Fast
python ./track_tuning.py -a MCCT -e ${fast_experts[@]} -m Fast

normal_experts=("DiMP" "DROL" "KYS" "PrDiMP" "RPT")
python ./track_tuning.py -e ${normal_experts[@]} -m Normal
python ./track_tuning.py -a HDT -e ${normal_experts[@]} -m Normal
python ./track_tuning.py -a MCCT -e ${normal_experts[@]} -m Normal

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
python ./track_tuning.py -e ${siamdw_experts[@]} -m SiamDW
python ./track_tuning.py -a HDT -e ${siamdw_experts[@]} -m SiamDW
python ./track_tuning.py -a MCCT -e ${siamdw_experts[@]} -m SiamDW

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
python ./track_tuning.py -e ${siamrpn_experts[@]} -m SiamRPN++
python ./track_tuning.py -a HDT -e ${siamrpn_experts[@]} -m SiamRPN++
python ./track_tuning.py -a MCCT -e ${siamrpn_experts[@]} -m SiamRPN++