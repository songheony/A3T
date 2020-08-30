# all_experts=("ATOM" "DaSiamRPN" "GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamMCF" "SiamRPN" "SiamRPN++" "SPM" "Staple" "THOR")
# python ./visualize_eval.py -e ${all_experts[@]} -d All

high_algorithm=("AAA_High_0.00_0.69_False_False_False_True_True_True_True")
high_baseline=("MCCT_High_0.10" "HDT_High_0.98" "Random_High" "Max_High")
high_experts=("ATOM" "DaSiamRPN" "SiamMCF" "SiamRPN++" "SPM" "THOR")
python ./visualize_eval.py -e ${high_experts[@]} -d High -a ${high_algorithm[@]} -b ${high_baseline[@]}

low_algorithm=("AAA_Low_0.00_0.60_False_False_False_True_True_True_True")
low_baseline=("MCCT_Low_0.10" "HDT_Low_0.32" "Random_Low" "Max_Low")
low_experts=("GradNet" "MemTrack" "SiamDW" "SiamFC" "SiamRPN" "Staple")
python ./visualize_eval.py -e ${low_experts[@]} -d Low -a ${low_algorithm[@]} -b ${low_baseline[@]}

mix_algorithm=("AAA_Mix_0.00_0.65_False_False_False_True_True_True_True")
mix_baseline=("MCCT_Mix_0.10" "HDT_Mix_0.94" "Random_Mix" "Max_Mix")
mix_experts=("ATOM" "SiamRPN++" "SPM" "MemTrack" "SiamFC" "Staple")
python ./visualize_eval.py -e ${mix_experts[@]} -d Mix -a ${mix_algorithm[@]} -b ${mix_baseline[@]}

siamdw_algorithm=("AAA_SiamDW_0.00_0.67_False_False_False_True_True_True_True")
siamdw_baseline=("MCCT_SiamDW_0.10" "HDT_SiamDW_0.98" "Random_SiamDW" "Max_SiamDW")
siamdw_experts=("SiamDW_SiamFCRes22" "SiamDW_SiamFCIncep22" "SiamDW_SiamFCNext22" "SiamDW_SiamRPNRes22" "SiamDW_SiamFCRes22_VOT" "SiamDW_SiamFCIncep22_VOT" "SiamDW_SiamFCNext22_VOT" "SiamDW_SiamRPNRes22_VOT")
python ./visualize_eval.py -e ${siamdw_experts[@]} -d SiamDW -a ${siamdw_algorithm[@]} -b ${siamdw_baseline[@]}

siamrpn_algorithm=("AAA_SiamRPN++_0.00_0.61_False_False_False_True_True_True_True")
siamrpn_baseline=("MCCT_SiamRPN++_0.10" "HDT_SiamRPN++_0.74" "Random_SiamRPN++" "Max_SiamRPN++")
siamrpn_experts=("SiamRPN++_AlexNet" "SiamRPN++_AlexNet_OTB" "SiamRPN++_ResNet-50" "SiamRPN++_ResNet-50_OTB" "SiamRPN++_ResNet-50_LT" "SiamRPN++_MobileNetV2" "SiamRPN++_SiamMask")
python ./visualize_eval.py -e ${siamrpn_experts[@]} -d SiamRPN++ -a ${siamrpn_algorithm[@]} -b ${siamrpn_baseline[@]}