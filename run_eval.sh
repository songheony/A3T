algorithmgood=("AAA_Good_0.00_0.69_False_False_False_True_True_True_True")
baselinegood=("Random_Good" "MCCT_Good" "Max_Good") 
goods=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
python ./visualize_eval.py -e ${goods[@]} -d Good -a ${algorithmgood[@]} -b ${baselinegood[@]}

algorithmbad=("AAA_Bad_0.00_0.60_False_False_False_True_True_True_True")
baselinerbad=("Random_Bad" "MCCT_Bad" "Max_Bad")
bads=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
python ./visualize_eval.py -e ${bads[@]} -d Bad -a ${algorithmbad[@]} -b ${baselinerbad[@]}

algorithmdw=("AAA_SiamDW_0.00_0.67_False_False_False_True_True_True_True")
baselinedw=("Random_SiamDW" "MCCT_SiamDW" "Max_SiamDW") 
siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
python ./visualize_eval.py -e ${siamdws[@]} -d SiamDW -a ${algorithmdw[@]} -b ${baselinedw[@]}

algorithmrpn=("AAA_SiamRPN++_0.00_0.61_False_False_False_True_True_True_True")
baselinerpn=("Random_SiamRPN++" "MCCT_SiamRPN++" "Max_SiamRPN++")
siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
python ./visualize_eval.py -e ${siamrpnpps[@]} -d SiamRPN++ -a ${algorithmrpn[@]} -b ${baselinerpn[@]}