import argparse
import re
from pathlib import Path
import os
import torch
from tqdm import tqdm

import sys
sys.path.append(".")
from inference.run_on_video import run_on_video


def process_one_folder(video_folder, mask_folder, output_folder):
    frames_with_masks = []
    for file_path in (p for p in Path(mask_folder).iterdir() if p.is_file()):
        frame_number_match = re.search(r'\d+', file_path.stem)
        if frame_number_match is None:
            print(f"ERROR: file {file_path} does not contain a frame number. Cannot load it as a mask.")
            exit(1)
        frames_with_masks.append(int(frame_number_match.group()))

    print("Using masks for frames: ", frames_with_masks)

    p_out = Path(output_folder)
    p_out.mkdir(parents=True, exist_ok=True)
    print("Processing {}".format(mask_folder))
    run_on_video(video_folder, mask_folder, output_folder, frames_with_masks,
                 print_progress=False,
                 image_saving_max_queue_size=400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video frames given a few (1+) existing annotation masks')
    parser.add_argument('--video', type=str,
                        help='Path to the video file or directory with .jpg video frames to process', required=False,
                        default="/home/ubuntu/XMem2/workspace/RefYTVOS-VideoLISA/videos")
    parser.add_argument('--masks', type=str,
                        help='Path to the directory with individual .png masks  for corresponding video frames, named `frame_000000.png`, `frame_000123.png`, ... or similarly (the script searches for the first integer value in the filename). '
                             'Will use all masks int the directory.', required=False,
                        default="/home/ubuntu/XMem2/workspace/RefYTVOS-VideoLISA/masks")
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    args = parser.parse_args()

    vid_root = args.video
    mask_root = args.masks
    output_root = args.masks + "_xmem_infer"
    os.makedirs(output_root, exist_ok=True)

    job_list_full = []

    vid_list = sorted(os.listdir(vid_root))
    assert vid_list == sorted(os.listdir(mask_root))
    for vid in vid_list:
        vid_folder = os.path.join(vid_root, vid)
        mask_folder = os.path.join(mask_root, vid)
        output_folder = os.path.join(output_root, vid)
        os.makedirs(output_folder, exist_ok=True)

        exp_list = sorted(os.listdir(mask_folder))
        for exp in exp_list:
            mask_folder_exp = os.path.join(mask_folder, exp)
            output_folder_exp = os.path.join(output_folder, exp)
            os.makedirs(output_folder, exist_ok=True)
            job_list_full.append((vid_folder, mask_folder_exp, output_folder_exp))

    job_list_subset = [job_list_full[i] for i in range(len(job_list_full)) if i % args.subset_num == args.subset_idx]
    progress_bar = tqdm(total=len(job_list_subset), desc='Progress {}'.format(args.subset_idx))
    for job in job_list_subset:
        process_one_folder(*job)
        torch.cuda.empty_cache()
        progress_bar.update(1)

