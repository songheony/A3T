datasets=("OTB2015" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT")

super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
for (( i=0; i<${#datasets[@]}; i++ )); do    
    python ./track_algorithm.py -a MCCT -m SuperFast -d ${datasets[$i]} -e ${super_fast_experts[@]} -t 0.1
done

fast_experts=("Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
for (( i=0; i<${#datasets[@]}; i++ )); do
        python ./track_algorithm.py -a MCCT -m Fast -d ${datasets[$i]} -e ${fast_experts[@]} -t 0.1
done

normal_experts=("DiMP" "DROL" "KYS" "PrDiMP" "RPT")
for (( i=0; i<${#datasets[@]}; i++ )); do
        python ./track_algorithm.py -a MCCT -m Normal -d ${datasets[$i]} -e ${normal_experts[@]} -t 0.1
done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
for (( i=0; i<${#datasets[@]}; i++ )); do
        python ./track_algorithm.py -a MCCT -m SiamDW -d ${datasets[$i]} -e ${siamdw_experts[@]} -f 0.1
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
for (( i=0; i<${#datasets[@]}; i++ )); do
        python ./track_algorithm.py -a MCCT -m SiamRPN++ -d ${datasets[$i]} -e ${siamrpn_experts[@]} -f 0.1
done