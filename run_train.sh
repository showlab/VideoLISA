# Reproduce the final result (Tab. 1, Tab. 2, and Tab. 3)
# The model is trained using 8 nodes (64 A10 GPUs), each A10 GPU has 24GB memory
deepspeed --hostfile hostfile_8nodes train_joint.py \
  --version="MBZUAI/LLaVA-Phi-3-mini-4k-instruct"  \
  --dataset_dir='/data_sdf/LLM_DATA/LISA/datasets' \
  --vision_pretrained="/home/ubuntu/ckpt/SAM/sam_vit_h_4b8939.pth" \
  --vision-tower="openai/clip-vit-large-patch14-336" \
  --exp_name="video-lisa-3.8b-6k-iter" \
  --num_frames_sparse=32 \
  --num_frames_dense=4 \
  --num_classes_per_sample=1 \
  --epochs=20 \
  --steps_per_epoch=300 \
  --batch_size=2 \
  --grad_accumulation_steps=1 \
  --model_max_length=2048 \
  --dataset="sem_seg,refer_seg,reason_seg,vos,refer_seg_video,davis" \
  --sem_seg_data="ade20k,cocostuff,pascal_part,paco_lvis" \
  --refer_seg_data="refclef,refcoco,refcoco+,refcocog" \
  --vos_data="ytvos" \
  --refer_seg_video_data="refer_youtube_vos,mevis" \
  --sample_rates="8,4,3,4,6,1"
