python 3-2_summarize_models.py \
    --proj_root ~/Alzheimers \
    --sweep_dir Results/ArchSweep \
    --hybrid_dirs "Results/Hybrid_Results_0826|hybrid_luke,Results/Hybrid_Results_0902|hybrid_luke+marco" \
    --hybrid_hp "hybrid_luke=~/Alzheimers/Results/Hybrid_Results_0826/Hyperparameters_hybrid_for_Luke.json,hybrid_luke+marco=~/Alzheimers/Results/Hybrid_Results_0902/Hyperparameters_hybrid_for_LukeMarco.json" \
    --families "ResNet,DenseNet,Inception,ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG,CNN_Small,Other" \
    --per_family 3 \
    --top_overall 100

# 
# After running, rename the output folder "Model_Leaderboard_Inc-Hybrids"
# if you want to run the single models only again. 
