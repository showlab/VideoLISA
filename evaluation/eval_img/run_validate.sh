
# ==================== ReasonSegVal dataset ===========================
deepspeed --master_port=24999 evaluation/eval_img/val.py \
  --version="ZechenBai/VideoLISA-3.8B" \
  --dataset_dir='/data_sdf/LLM_DATA/LISA/datasets' \
  --vision_pretrained="/home/ubuntu/ckpt/SAM/sam_vit_h_4b8939.pth" \
  --vision_tower="openai/clip-vit-large-patch14-336" \
  --num_frames_sparse=32 \
  --num_frames_dense=4 \
  --model_max_length=2048 \
  --eval_only \
  --val_dataset="ReasonSeg|val"

# ReasonSeg|val, ReasonSeg|test|short, ReasonSeg|val|long, ReasonSeg|val|all
# refcoco|unc|testA, refcoco|unc|testB, refcoco+|unc|testA, ...

