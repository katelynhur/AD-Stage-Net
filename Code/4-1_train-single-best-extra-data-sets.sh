python 3-1_multi_arch_mini_sweep.py \
  --proj_root ~/Alzheimers \
  --train_dir Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/train \
  --test_dir  Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --results_subdir Results/Singles_Luke \
  --epochs 20 \
  --best_params_json Results/Model_Leaderboard/best_retrained.json \
  --single_best_only 1 \
  --extra_tests "Marco:Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/test;Falah:Data/HuggingFace_Falah_Alzheimer_MRI/test" \
  --gpus auto

python 3-1_multi_arch_mini_sweep.py \
  --proj_root ~/Alzheimers \
  --train_dir Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/train \
  --test_dir  Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/test \
  --results_subdir Results/Singles_Marco \
  --epochs 20 \
  --best_params_json Results/Model_Leaderboard/best_retrained.json \
  --single_best_only 1 \
  --extra_tests "Luke:Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test;Falah:Data/HuggingFace_Falah_Alzheimer_MRI/test" \
  --gpus auto

python 3-1_multi_arch_mini_sweep.py \
  --proj_root ~/Alzheimers \
  --train_dir Data/HuggingFace_Falah_Alzheimer_MRI/train \
  --test_dir  Data/HuggingFace_Falah_Alzheimer_MRI/test \
  --results_subdir Results/Singles_Falah \
  --epochs 20 \
  --best_params_json Results/Model_Leaderboard/best_retrained.json \
  --single_best_only 1 \
  --extra_tests "Luke:Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test;Marco:Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/test" \
  --gpus auto

