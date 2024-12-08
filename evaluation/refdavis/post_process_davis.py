import os
import json
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, required=True)
parser.add_argument("--davis_path", type=str,
                    default="/data_sdf/LLM_DATA/video_centric/ref-davis")
args = parser.parse_args()

src_dir = args.src_dir
dst_dir = src_dir + '_processed'
davis_path = args.davis_path

# get palette
palette_img = os.path.join(davis_path, "valid/Annotations/blackswan/00000.png")
palette = Image.open(palette_img).getpalette()

meta_exp_path = os.path.join(davis_path, "meta_expressions/valid/meta_expressions.json")
data = json.load(open(meta_exp_path, 'r'))["videos"]
video_list = list(data.keys())

for video in tqdm(video_list):
    metas = []

    expressions = data[video]["expressions"]
    expression_list = list(expressions.keys())
    num_expressions = len(expression_list)
    video_len = len(data[video]["frames"])

    # read all the anno meta
    for i in range(num_expressions):
        meta = {}
        meta["video"] = video
        meta["exp"] = expressions[expression_list[i]]["exp"]
        meta["exp_id"] = expression_list[i]  # start from 0
        meta["frames"] = data[video]["frames"]
        metas.append(meta)
    meta = metas

    # since there are 4 annotations
    num_obj = num_expressions // 4

    # 2. for each annotator
    for anno_id in range(4):  # 4 annotators
        anno_logits = []
        anno_masks = []  # [num_obj+1, video_len, h, w], +1 for background

        for obj_id in range(num_obj):
            i = obj_id * 4 + anno_id
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)

            mask_dir = os.path.join(src_dir, video_name, exp_id)
            mask_file_list = sorted(os.listdir(mask_dir))
            assert video_len == len(mask_file_list)

            all_pred_masks = []
            for mask_file in mask_file_list:
                pred_mask_np = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
                pred_mask_tsr = torch.from_numpy(pred_mask_np) / 100.0
                all_pred_masks.append(pred_mask_tsr)
            all_pred_masks = torch.stack(all_pred_masks, dim=0)     # (video_len, h, w)

            anno_masks.append(all_pred_masks)

        anno_masks = torch.stack(anno_masks)  # [num_obj, video_len, h, w]
        t, h, w = anno_masks.shape[-3:]
        anno_masks[anno_masks < 0.5] = 0.0
        background = 0.1 * torch.ones(1, t, h, w)
        anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, video_len, h, w]
        out_masks = torch.argmax(anno_masks, dim=0)  # int, the value indicate which object, [video_len, h, w]

        out_masks = out_masks.detach().cpu().numpy().astype(np.uint8)  # [video_len, h, w]

        # save results
        anno_save_path = os.path.join(dst_dir, f"anno_{anno_id}", video)
        if not os.path.exists(anno_save_path):
            os.makedirs(anno_save_path)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))

