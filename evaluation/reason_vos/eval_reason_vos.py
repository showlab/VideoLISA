###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import os
import time
import argparse
import cv2
import json
import glob
import numpy as np
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp

NUM_WOEKERS = 64


def eval_queue(q, rank, out_dict, mask_path, pred_path):
    while not q.empty():
        src_dataset, vid_name, exp_id = q.get()
        pred_path_vid_exp = os.path.join(pred_path, vid_name, exp_id)

        vid_key = "{}_{}_{}".format(src_dataset, vid_name, exp_id)
        mask_path_vid = os.path.join(mask_path, vid_key)

        if not os.path.exists(mask_path_vid):
            print(f'{mask_path_vid} not found, not take into metric computation')
            continue
        if not os.path.exists(pred_path_vid_exp):
            print(f'{pred_path_vid_exp} not found, not take into metric computation')
            continue

        gt_mask_list = [x for x in sorted(os.listdir(mask_path_vid)) if str(x).endswith('png')]
        gt_0_path = os.path.join(mask_path_vid, gt_mask_list[0])
        gt_0 = cv2.imread(gt_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt_0.shape

        vid_len = len(gt_mask_list)
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        for frame_idx, frame_name in enumerate(gt_mask_list):
            gt_masks[frame_idx] = cv2.imread(os.path.join(mask_path_vid, frame_name), cv2.IMREAD_GRAYSCALE)
            pred_masks[frame_idx] = cv2.imread(os.path.join(pred_path_vid_exp, frame_name), cv2.IMREAD_GRAYSCALE)

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[vid_key] = [j, f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_exp_path", type=str, default="ReasonVOS/meta_expressions.json")
    parser.add_argument("--mask_path", type=str, default="ReasonVOS/Annotations")
    parser.add_argument("--pred_path", type=str, default="")
    parser.add_argument("--save_name", type=str, default="")
    args = parser.parse_args()

    queue = mp.Queue()
    meta_exp = json.load(open(args.meta_exp_path, 'r'))["videos"]
    output_dict = mp.Manager().dict()

    for vid_name in meta_exp.keys():
        vid = meta_exp[vid_name]
        src_dataset = vid['source']
        is_sent = vid['is_sent']
        for exp in vid['expressions']:
            exp_id = vid['expressions'][exp]['obj_id']
            queue.put([src_dataset, vid_name, str(exp_id)])

    print("Q-Size:", queue.qsize())

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.mask_path, args.pred_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))

