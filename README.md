<div align="center">
<br>
<h3>One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos</h3>

[Zechen Bai](https://www.baizechen.site/) <sup>1</sup>&nbsp;
[Tong He](https://hetong007.github.io/) <sup>2</sup>&nbsp;
[Haiyang Mei](https://mhaiyang.github.io/) <sup>1</sup>&nbsp;
[Pichao Wang](https://wangpichao.github.io/) <sup>2</sup>&nbsp;
[Ziteng Gao](https://sebgao.github.io/) <sup>1</sup>&nbsp;
[Joya Chen](https://chenjoya.github.io/) <sup>1</sup>&nbsp;
[Lei Liu](https://openreview.net/profile?id=~liulei2) <sup>2</sup>&nbsp;
[Zheng Zhang](https://scholar.google.com/citations?user=k0KiE4wAAAAJ&hl=en) <sup>2</sup>&nbsp;
[Mike Zheng Shou](https://sites.google.com/view/showlab) <sup>1</sup>&nbsp;

NeurIPS 2024

<sup>1</sup> [Show Lab, National University of Singapore](https://sites.google.com/view/showlab/home?authuser=0) &nbsp; <sup>2</sup> Amazon&nbsp;
 
[![arXiv](https://img.shields.io/badge/arXiv-<2409.19603>-<COLOR>.svg)](https://arxiv.org/abs/2409.19603)

</div>

**News**
* **[2024-11-26]** We released pre-trained VideoLISA-3.8B at [HuggingFace](https://huggingface.co/ZechenBai/VideoLISA-3.8B)!.
* **[2024-11-20]** We released the training and inference code.
* **[2024-09-29]** We released our paper on [arXiv](https://arxiv.org/abs/2409.19603).

<p align="center"> <img src="assets/framework.jpg" width="666"></p>

<p align="center"> <img src="assets/teaser.jpg" width="666"></p>

## TODO
- [X] Release the inference code.
- [X] Release the training code.
- [ ] Instructions on supporting more datasets.

## Setup Environment
```shell
conda create -n videolisa python=3.10 -y
conda activate videolisa
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

## Prepare Data

First, please prepare the image data following this [instruction in LISA](https://github.com/dvlab-research/LISA/tree/main?tab=readme-ov-file#training-data-preparation).

We introduce the video datasets used in this project.
Note that the data paths for video datasets are currently hard-coded in each dataset file in the `utils` folder.
You may need to adjust it accordingly.

### MeViS
Download the dataset from the [official release](https://github.com/henghuiding/MeViS).
Then, extract and organize the file. We expect the directory structure to be the following:

```
mevis
├── train                       // Split Train
│   ├── JPEGImages
│   │   ├── <video #1  >
│   │   ├── <video #2  >
│   │   └── <video #...>
│   │
│   ├── mask_dict.json
│   └── meta_expressions.json
│
├── valid_u                     // Split Val^u
│   ├── JPEGImages
│   │   └── <video ...>
│   │
│   ├── mask_dict.json
│   └── meta_expressions.json
│
└── valid                       // Split Val
    ├── JPEGImages
    │   └── <video ...>
    │
    └── meta_expressions.json
```


### Ref-YouTube-VOS and Ref-DAVIS-17
Prepare Ref-YouTube-VOS and Ref-DAVIS-17 datasets following the instructions of [ReferFormer](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md).

### YouTube-VOS
Download teh dataset from the [website](https://youtube-vos.org/dataset/vos/) and organize it as follows:
```
YTVOS
├── train
│   ├── JPEGImages
│   ├── Annotations
│   ├── meta.json
```

## Training
We provide a sample training script in `run_train.sh`.
In our own experiments, we use 8 node (64 A10 24G GPUs) in total to train the model.
Under this setting, we set `batch_size=2` and `grad_accumulation_steps=1`,
so that the global effective batch size is `batch_size*grad_accumulation_steps*num_gpus=128`.
You can modify these settings based on your hardwares.
However, we did not explore other training hyper-parameters.
If you don't have sufficient GPUs, don't give up, you may still try to train the model with small batch size.
One tip: if you use small batch size, also reducing the learning rate might help.

After training finished, to get the full model weight:
```shell
cd ./runs/video-lisa-3.8b-3k-iter/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### Weight merging
Since the script do LoRA training with the help of deepspeed by default, after training, you need to merge the lora weights back to the model.
```shell
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="MBZUAI/LLaVA-Phi-3-mini-4k-instruct" \
  --weight="runs/video-lisa-3.8b-3k-iter/pytorch_model.bin" \
  --save_path="runs/video-lisa-3.8b-3k-iter/merged"
```

## Evaluation

### MeViS

Before jumping into the follow commands, you may look into the involved scripts and config the data paths.
```shell
# Step 1
bash evaluation/mevis_val_u/run_inference_mevis.sh

# Step 2
bash evaluation/mevis_val_u/run_eval_mevis.sh
```

### Other Datasets
Ongoing.


### Citation
To cite the paper and model, please use the below:
```
@article{bai2024videolisa,
  title={One token to seg them all: Language instructed reasoning segmentation in videos},
  author={Bai, Zechen and He, Tong and Mei, Haiyang and Wang, Pichao and Gao, Ziteng and Chen, Joya and Liu, Lei and Zhang, Zheng and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2409.19603},
  year={2024}
}
```
### Acknowledgments
This work is heavily based on [LISA](https://github.com/dvlab-research/LISA/), [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-pp](https://github.com/mbzuai-oryx/LLaVA-pp), [Segment-Anything](https://github.com/facebookresearch/segment-anything) and [Phi-3](https://github.com/microsoft/Phi-3CookBook). Thanks to all the authors for their great works!
