VIS_SAVE_PATH="results/VideoLISA-MeViS-Valid-U/"   # remember to add /

# Step-2: run evaluation
python evaluation/mevis_val_u/eval_mevis.py \
  --mevis_pred_path $VIS_SAVE_PATH \
  --save_name $VIS_SAVE_PATH"result.json"

