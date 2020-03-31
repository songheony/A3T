datasets=("OTB" "NFS" "UAV" "TPL" "VOT" "LaSOT")

all=("ATOM" "DaSiamRPN" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
thresholds=(0.62)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n All -e ${all[@]} -r ${thresholds[$i]}
    done
done

goods=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
thresholds=(0.69)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n Good -e ${goods[@]} -r ${thresholds[$i]}
    done
done

bads=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
thresholds=(0.60)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n Bad -e ${bads[@]} -r ${thresholds[$i]}
    done
done

mixs=("ATOM" "SiamRPN++" "SPM" "MemTrack" "SiamFC" "Staple")
thresholds=(0.65)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n Mix -e ${mixs[@]} -r ${thresholds[$i]}
    done
done

siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
thresholds=(0.67)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n SiamDW -e ${siamdws[@]} -r ${thresholds[$i]}
    done
done

siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
thresholds=(0.61)
for (( j=0; j<${#datasets[@]}; j++ )); do    
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        python ./track_algorithm.py -a AAA -d ${datasets[$j]} -n SiamRPN++ -e ${siamrpnpps[@]} -r ${thresholds[$i]}
    done
done