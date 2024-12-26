import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np
from glob import glob


def uniform_sample(total_len, sample_num):
    intervals = np.linspace(start=0, stop=total_len, num=sample_num + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def get_sparse_indices(total_frame_num, num_frames_sparse):
    if total_frame_num > num_frames_sparse:       # video is long, uniformly sample frames
        frame_idxs = uniform_sample(total_frame_num, num_frames_sparse)
        return sorted(frame_idxs)
    else:
        num_repeat = num_frames_sparse // total_frame_num
        num_sample = num_frames_sparse % total_frame_num
        frame_idxs = list(range(total_frame_num)) * num_repeat + uniform_sample(total_frame_num, num_sample)
        return sorted(frame_idxs)


def get_dense_indices(num_frames_sparse, num_frames_dense):
    intervals = np.linspace(start=0, stop=num_frames_sparse - 1, num=num_frames_dense + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=str, help='Path to the folder with video frames', required=False,
                        default="/home/ubuntu/XMem2/refytvos_custom/JPEGImages")
    parser.add_argument('--masks', type=str, help='Path to the folder with existing masks', required=False,
                        default="/home/ubuntu/VideoLISA-Phi3/evaluation/refytvos/results")
    parser.add_argument('--dst_dir', type=str, help='Path to the folder with existing masks', required=False,
                        default="/home/ubuntu/XMem2/mevis_valid_u_custom/Annotations-VideoLISAPhi")
    parser.add_argument("--num_frames_sparse", default=32, type=int)
    parser.add_argument("--num_frames_dense", default=4, type=int)
    args = parser.parse_args()

    vid_path_input = args.videos
    vid_list = sorted(os.listdir(vid_path_input))

    mask_vid_path_input = args.masks
    mask_vid_list = [x for x in sorted(os.listdir(mask_vid_path_input)) if not str(x).endswith('json')]

    mask_vid_path_output = args.dst_dir
    os.makedirs(mask_vid_path_output, exist_ok=True)

    print("Will save the selected masks into {}.".format(mask_vid_path_output))

    assert vid_list == mask_vid_list

    for vid_name, mask_vid_name in tqdm(zip(vid_list, mask_vid_list)):
        assert vid_name == mask_vid_name

        # prepare video frames
        image_folder = os.path.join(vid_path_input, vid_name)
        if not os.path.exists(image_folder):
            print("File not found in {}".format(image_folder))
            raise FileNotFoundError

        image_file_list = sorted(glob(os.path.join(image_folder, '*.jpg')))
        total_frames = len(image_file_list)

        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_sparse)
        dense_idxs = get_dense_indices(args.num_frames_sparse, args.num_frames_dense)
        frame_idxs = [sparse_idxs[x] for x in dense_idxs]

        for exp_id in sorted(os.listdir(os.path.join(mask_vid_path_input, mask_vid_name))):
            exp_src_dir = os.path.join(mask_vid_path_input, mask_vid_name, exp_id)
            exp_save_dir = os.path.join(mask_vid_path_output, mask_vid_name, exp_id)
            os.makedirs(exp_save_dir, exist_ok=True)

            mask_file_list = sorted(os.listdir(exp_src_dir))
            for frm_idx in frame_idxs:
                src_file = os.path.join(exp_src_dir, mask_file_list[frm_idx])
                dst_file = os.path.join(exp_save_dir, mask_file_list[frm_idx])
                shutil.copyfile(src_file, dst_file)

