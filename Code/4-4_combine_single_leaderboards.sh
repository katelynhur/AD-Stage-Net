python 4-4_combine_single_leaderboards.py \
  --proj_root ~/Alzheimers \
  --inputs "Luke=Results/Model_Leaderboard_Singles_Luke/leaderboard.csv,Marco=Results/Model_Leaderboard_Singles_Marco/leaderboard.csv,Falah=Results/Model_Leaderboard_Singles_Falah/leaderboard.csv" \
  --out_dir Results/Combined_Singles_Leaderboard \
  --join_on arch \
  --top_n 20

