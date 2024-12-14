VIS_SAVE_PATH="results/VideoLISA-ReasonVOS/"   # remember to add /


python evaluation/reason_vos/eval_reason_vos.py \
  --pred_path $VIS_SAVE_PATH \
  --save_name $VIS_SAVE_PATH"result.json"

