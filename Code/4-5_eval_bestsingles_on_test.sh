# (Optional) Luke test set → should match arch-sweep accuracy
python 4-5_eval_bestsingles_on_test.py \
  --proj_root ~/Alzheimers \
  --weights_root Results/BestSingles \
  --test_dir Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --out_subdir Results/BestSingles_Luke \
  --sweep_csv Results/Singles_Luke/arch_sweep_results.csv

# Marco test set → Results/BestSingles_Marco
python 4-5_eval_bestsingles_on_test.py \
  --proj_root ~/Alzheimers \
  --weights_root Results/BestSingles \
  --test_dir Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/test \
  --out_subdir Results/BestSingles_Marco \
  --sweep_csv Results/Singles_Luke/arch_sweep_results.csv

# Falah test set → Results/BestSingles_Falah
python 4-5_eval_bestsingles_on_test.py \
  --proj_root ~/Alzheimers \
  --weights_root Results/BestSingles \
  --test_dir Data/HuggingFace_Falah_Alzheimer_MRI/test \
  --out_subdir Results/BestSingles_Falah \
  --sweep_csv Results/Singles_Luke/arch_sweep_results.csv
