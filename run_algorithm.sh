datasets=("OTB2015" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT")

super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
for (( i=0; i<${#datasets[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d ${datasets[$i]} -m SuperFast -e ${super_fast_experts[@]} -t 0.70
done

fast_experts=("Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
for (( i=0; i<${#datasets[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d ${datasets[$i]} -m Fast -e ${fast_experts[@]} -t 0.72
done

normal_experts=("DiMP" "DROL" "KYS" "PrDiMP" "RPT")
for (( i=0; i<${#datasets[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d ${datasets[$i]} -m Normal -e ${normal_experts[@]} -t 0.74
done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
for (( i=0; i<${#datasets[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d ${datasets[$i]} -m SiamDW -e ${siamdw_experts[@]} -t 0.73
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
for (( i=0; i<${#datasets[@]}; i++ )); do    
    python ./track_algorithm.py -a AAA -d ${datasets[$i]} -m SiamRPN++ -e ${siamrpn_experts[@]} -t 0.75
done
