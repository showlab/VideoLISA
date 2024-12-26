# ========================
VIDEO_PATH=/home/ubuntu/XMem2/refytvos_custom/JPEGImages
RAW_MASK_PATH=/home/ubuntu/VideoLISA-Phi3/evaluation/refytvos/results
WORKSPACE_NAME=RefYTVOS-VideoLISA
SELECT_MASK_PATH=/home/ubuntu/XMem2/refytvos_custom/Annotations-VideoLISAPhi
NUM_FRMS_DENSE=4
NUM_FRMS_SPARSE=32
INFERENCE_SCRIPT=xmem2_example/process_video.py
SUBSET_NUM=8


# ======== Step 1: select effective mask produced by [TRK] token ============
python xmem2_example/prepare_files.py \
  --videos $VIDEO_PATH \
  --masks $RAW_MASK_PATH \
  --dst_dir $SELECT_MASK_PATH \
  --num_frames_sparse $NUM_FRMS_SPARSE \
  --num_frames_dense $NUM_FRMS_DENSE


# ======== Step 2: import the data into the workspace of XMem2 ============
python xmem2_example/import_proj.py \
  --name $WORKSPACE_NAME\
  --size 0 \
  --videos $VIDEO_PATH \
  --masks $SELECT_MASK_PATH


# ======== Step 3: run inference ==========
CUDA_VISIBLE_DEVICES=0 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 0 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=1 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 1 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=2 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 2 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=3 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 3 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=4 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 4 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=5 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 5 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=6 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 6 \
  --subset_num $SUBSET_NUM &
CUDA_VISIBLE_DEVICES=7 python $INFERENCE_SCRIPT \
  --video "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/videos" \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks" \
  --subset_idx 7 \
  --subset_num $SUBSET_NUM


# ======== Step 4: recover color platte ==========
python xmem2_example/recover.py \
  --video $VIDEO_PATH \
  --masks "/home/ubuntu/XMem2/workspace/"$WORKSPACE_NAME"/masks"
