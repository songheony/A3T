algorithmall=("AAA_All_0.00_0.66_False_False_False_True_True_True_True")
baselineall=("Random_All" "MCCT_All" "Max_All") 
all=("ATOM" "DaSiamRPN" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
python ./visualize_eval.py -e ${all[@]} -d All -a ${algorithmall[@]} -b ${baselineall[@]}

algorithmgood=("AAA_Good_0.00_0.69_False_False_False_True_True_True_True")
baselinegood=("Random_Good" "MCCT_Good" "Max_Good") 
goods=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
python ./visualize_eval.py -e ${goods[@]} -d Good -a ${algorithmgood[@]} -b ${baselinegood[@]}

algorithmbad=("AAA_Bad_0.00_0.60_False_False_False_True_True_True_True")
baselinerbad=("Random_Bad" "MCCT_Bad" "Max_Bad")
bads=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
python ./visualize_eval.py -e ${bads[@]} -d Bad -a ${algorithmbad[@]} -b ${baselinerbad[@]}

algorithmmix=("AAA_Mix_0.00_0.65_False_False_False_True_True_True_True")
baselinermix=("Random_Mix" "MCCT_Mix" "Max_Mix")
mixs=("ATOM" "SiamRPN++" "SPM" "MemTrack" "SiamFC" "Staple")
python ./visualize_eval.py -e ${mixs[@]} -d Mix -a ${algorithmmix[@]} -b ${baselinermix[@]}

# algorithmdw=("AAA_SiamDW_0.00_0.67_False_False_False_True_True_True_True")
# baselinedw=("Random_SiamDW" "MCCT_SiamDW" "Max_SiamDW") 
# siamdws=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
# python ./visualize_eval.py -e ${siamdws[@]} -d SiamDW -a ${algorithmdw[@]} -b ${baselinedw[@]}

algorithmrpn=("AAA_SiamRPN++_0.00_0.61_False_False_False_True_True_True_True")
baselinerpn=("Random_SiamRPN++" "MCCT_SiamRPN++" "Max_SiamRPN++")
siamrpnpps=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
python ./visualize_eval.py -e ${siamrpnpps[@]} -d SiamRPN++ -a ${algorithmrpn[@]} -b ${baselinerpn[@]}