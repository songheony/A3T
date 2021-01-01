datasets=("OTB2015" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT")

super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
threshold=0.1
for (( j=0; j<${#datasets[@]}; j++ )); do    
    python ./track_algorithm.py -a MCCT -n SuperFast -d ${datasets[$j]} -e ${super_fast_experts[@]} -r $threshold
done

fast_experts=("Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
threshold=0.1
for (( j=0; j<${#datasets[@]}; j++ )); do
        python ./track_algorithm.py -a MCCT -n Fast -d ${datasets[$j]} -e ${fast_experts[@]} -r $threshold
done

normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "SiamMCF")
threshold=0.1
for (( j=0; j<${#datasets[@]}; j++ )); do
        python ./track_algorithm.py -a MCCT -n Normal -d ${datasets[$j]} -e ${normal_experts[@]} -r $threshold
done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
threshold=0.1
for (( j=0; j<${#datasets[@]}; j++ )); do
        python ./track_algorithm.py -a MCCT -n SiamDW -d ${datasets[$j]} -e ${siamdw_experts[@]} -r $threshold
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
threshold=0.1
for (( j=0; j<${#datasets[@]}; j++ )); do
        python ./track_algorithm.py -a MCCT -n SiamRPN++ -d ${datasets[$j]} -e ${siamrpn_experts[@]} -r $threshold
done