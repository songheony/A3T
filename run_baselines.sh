datasets=("OTB2015" "OTB2015-80%" "OTB2015-60%" "OTB2015-40%" "OTB2015-20%" "NFS" "UAV123" "TColor128" "VOT2018" "LaSOT" "Got10K")
baselines=("Random" "Max", "Without delay")

super_fast_experts=("DaSiamRPN" "SiamDW" "SiamRPN" "SPM")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n SuperFast -d ${datasets[$j]} -e ${super_fast_experts[@]}
    done
done

fast_experts=("GradNet" "Ocean" "SiamBAN" "SiamCAR" "SiamFC++" "SiamRPN++")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n Fast -d ${datasets[$j]} -e ${fast_experts[@]}
    done
done

normal_experts=("ATOM" "DiMP" "DROL" "KYS" "PrDiMP" "SiamMCF")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n Normal -d ${datasets[$j]} -e ${normal_experts[@]}
    done
done

siamdw_experts=("SiamDWGroup/SiamFCRes22/OTB" "SiamDWGroup/SiamFCIncep22/OTB" "SiamDWGroup/SiamFCNext22/OTB" "SiamDWGroup/SiamRPNRes22/OTB" "SiamDWGroup/SiamFCRes22/VOT" "SiamDWGroup/SiamFCIncep22/VOT" "SiamDWGroup/SiamFCNext22/VOT" "SiamDWGroup/SiamRPNRes22/VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n SiamDW -d ${datasets[$j]} -e ${siamdw_experts[@]}
    done
done

siamrpn_experts=("SiamRPN++Group/AlexNet/VOT" "SiamRPN++Group/AlexNet/OTB" "SiamRPN++Group/ResNet-50/VOT" "SiamRPN++Group/ResNet-50/OTB" "SiamRPN++Group/ResNet-50/VOTLT" "SiamRPN++Group/MobileNetV2/VOT" "SiamRPN++Group/SiamMask/VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n SiamRPN++ -d ${datasets[$j]} -e ${siamrpn_experts[@]}
    done
done