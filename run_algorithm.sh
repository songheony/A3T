datasets=("OTB2015" "OTB2015-80%" "OTB2015-60%" "OTB2015-40%" "OTB2015-20%" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT" "Got10K")

super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
thresholds=(0.69)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n SuperFast -e ${super_fast_experts[@]} -r ${thresholds[$i]}
    done
done

fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
thresholds=(0.60)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n Fast -e ${fast_experts[@]} -r ${thresholds[$i]}
    done
done

normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "SiamMCF")
thresholds=(0.65)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n Normal -e ${normal_experts[@]} -r ${thresholds[$i]}
    done
done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
thresholds=(0.67)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n SiamDW -e ${siamdw_experts[@]} -r ${thresholds[$i]}
    done
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
thresholds=(0.61)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n SiamRPN++ -e ${siamrpn_experts[@]} -r ${thresholds[$i]}
    done
done