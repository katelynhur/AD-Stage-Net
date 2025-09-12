python 3-1_multi_arch_mini_sweep.py \
  --proj_root ~/Alzheimers \
  --train_dir Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/train \
  --test_dir  Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --results_subdir Results/ArchSweep \
  --epochs 20 \
  --batch_size 32 \
  --best_params_json ~/Alzheimers/Results/HP_Search_Luke_DDP/best_params.json \
  --gpus auto \
  --skip_completed 1
