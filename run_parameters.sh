thresholds=(0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79)

super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
for (( i=0; i<${#thresholds[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d OTB2015 -m SuperFast -e ${super_fast_experts[@]} -t ${thresholds[$i]}
done

fast_experts=("Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
for (( i=0; i<${#thresholds[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d OTB2015 -m Fast -e ${fast_experts[@]} -t ${thresholds[$i]}
done

normal_experts=("DiMP" "DROL" "KYS" "PrDiMP" "RPT")
for (( i=0; i<${#thresholds[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d OTB2015 -m Normal -e ${normal_experts[@]} -t ${thresholds[$i]}
done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
for (( i=0; i<${#thresholds[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d OTB2015 -m SiamDW -e ${siamdw_experts[@]} -t ${thresholds[$i]}
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
for (( i=0; i<${#thresholds[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d OTB2015 -m SiamRPN++ -e ${siamrpn_experts[@]} -t ${thresholds[$i]}
done
