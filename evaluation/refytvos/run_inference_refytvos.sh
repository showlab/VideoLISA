SCRIPT="evaluation/refytvos/inference_refytvos.py"
VERSION="ZechenBai/VideoLISA-3.8B"
VID_TOWER="openai/clip-vit-large-patch14-336"
NUM_FRMS_SPARSE=32
NUM_FRMS_DENSE=4
SUBSET_NUM=8
VIS_SAVE_PATH="results/VideoLISA-RefYTVOS"

# Step-1: run inference
CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
  --subset_idx=0 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM &
CUDA_VISIBLE_DEVICES=1 python $SCRIPT \
  --subset_idx=1 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM &
CUDA_VISIBLE_DEVICES=2 python $SCRIPT \
  --subset_idx=2 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM &
CUDA_VISIBLE_DEVICES=3 python $SCRIPT \
  --subset_idx=3 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM  &
CUDA_VISIBLE_DEVICES=4 python $SCRIPT \
  --subset_idx=4 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM &
CUDA_VISIBLE_DEVICES=5 python $SCRIPT \
  --subset_idx=5 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM &
CUDA_VISIBLE_DEVICES=6 python $SCRIPT \
  --subset_idx=6 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM &
CUDA_VISIBLE_DEVICES=7 python $SCRIPT \
  --subset_idx=7 \
  --version=$VERSION \
  --vision_tower=$VID_TOWER \
  --vis_save_path=$VIS_SAVE_PATH \
  --num_frames_dense=$NUM_FRMS_DENSE \
  --num_frames_sparse=$NUM_FRMS_SPARSE \
  --subset_num=$SUBSET_NUM

