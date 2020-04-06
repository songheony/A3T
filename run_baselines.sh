datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")
baselines=("MCCT" "Random" "Max")

# all=("ATOM" "DaSiamRPN" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
# for (( j=0; j<${#datasets[@]}; j++ )); do    
#     for (( i=0; i<${#baselines[@]}; i++ )); do
#         python ./track_algorithm.py -a ${baselines[$i]} -n All -d ${datasets[$j]} -e ${all[@]}
#     done
# done

goods=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n Good -d ${datasets[$j]} -e ${goods[@]}
    done
done

bads=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n Bad -d ${datasets[$j]} -e ${bads[@]}
    done
done

mixs=("ATOM" "SiamRPN++" "SPM" "MemTrack" "SiamFC" "Staple")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n Mix -d ${datasets[$j]} -e ${mixs[@]}
    done
done

siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n SiamDW -d ${datasets[$j]} -e ${siamdws[@]}
    done
done

siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#baselines[@]}; i++ )); do
        python ./track_algorithm.py -a ${baselines[$i]} -n SiamRPN++ -d ${datasets[$j]} -e ${siamrpnpps[@]}
    done
done