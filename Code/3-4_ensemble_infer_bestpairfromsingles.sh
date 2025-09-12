python 3-4_ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --candidates_json ~/Alzheimers/Results/Model_Leaderboard/top_for_hybrids.json \
  --include_families "ResNet,DenseNet,Inception,ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG" \
  --limit_per_family 3 \
  --search pair \
  --one_per_family 1 \
  --out_dir ~/Alzheimers/Results/EnsembleEval/Pairs_LukeTest_From_Luke

python 3-4_ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/test \
  --candidates_json ~/Alzheimers/Results/Model_Leaderboard/top_for_hybrids.json \
  --include_families "ResNet,DenseNet,Inception,ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG" \
  --limit_per_family 3 \
  --search pair \
  --one_per_family 1 \
  --out_dir ~/Alzheimers/Results/EnsembleEval/Pairs_MarcoTest_From_Luke

python 3-4_ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/HuggingFace_Falah_Alzheimer_MRI/test \
  --candidates_json ~/Alzheimers/Results/Model_Leaderboard/top_for_hybrids.json \
  --include_families "ResNet,DenseNet,Inception,ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG" \
  --limit_per_family 3 \
  --search pair \
  --one_per_family 1 \
  --out_dir ~/Alzheimers/Results/EnsembleEval/Pairs_FalahTest_From_Luke

