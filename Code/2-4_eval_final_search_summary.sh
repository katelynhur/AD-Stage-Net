python 2-4_eval_final_search_summary.py \
  --proj_root ~/Alzheimers \
  --checkpoint Results/HP_Search_Luke_DDP/final_best_resnet50.pt \
  --data_dir Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --hpo_csv Results/HP_Search_Luke_DDP/resnet50_luke_small_search_ddp.csv \
  --retrain_csv Results/HP_Search_Luke_DDP/retrain_summary_mean_std.csv

# Optional with specific override
# e.g., different eval batch or non-ResNet size
#python 2-4_eval_final_search_summary.py \
#  --proj_root ~/Alzheimers \
#  --checkpoint Results/HP_Search_Luke_DDP/final_best_resnet50.pt \
#  --data_dir Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
#  --img_size 224 \
#  --batch_size 64 \
#  --dropout 0.3
#
#
