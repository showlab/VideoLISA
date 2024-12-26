import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import multiprocessing


def sanity_check(vid_root, mask_root):
    output_root = mask_root + "_xmem_infer"

    vid_list = sorted(os.listdir(vid_root))
    assert vid_list == sorted(os.listdir(mask_root)) == sorted(os.listdir(output_root))

    for vid in tqdm(vid_list):
        vid_folder = os.path.join(vid_root, vid)
        mask_folder = os.path.join(mask_root, vid)
        output_folder = os.path.join(output_root, vid)

        num_imgs = len(os.listdir(vid_folder))
        exp_list = sorted(os.listdir(mask_folder))
        assert exp_list == sorted(os.listdir(output_folder)), (sorted(os.listdir(output_folder)), exp_list)
        for exp in exp_list:
            output_folder_exp = os.path.join(output_folder, exp, "masks")
            if not os.path.exists(output_folder_exp):
                print(output_folder_exp)
                continue
            else:
                assert num_imgs == len(os.listdir(output_folder_exp)), (num_imgs, output_folder_exp)

    print("Sanity check passed.")


def segmentation_mask_to_binary(segmentation_mask_path, reference_size, output_mask_path):
    # Load segmentation mask
    segmentation_mask = Image.open(segmentation_mask_path)

    # Resize segmentation mask to match the size of the reference image
    resized_mask = segmentation_mask.resize(reference_size, Image.NEAREST)

    # Convert segmentation mask to numpy array
    segmentation_array = np.array(resized_mask)

    # Extract unique colors from the segmentation mask
    unique_colors = np.unique(segmentation_array.reshape(-1, segmentation_array.shape[2]), axis=0)

    # Assuming there's only one object, pick the first unique color
    object_color = unique_colors[0]

    # Create binary mask by checking equality with the object color
    binary_mask = np.all(segmentation_array == object_color, axis=-1).astype(np.uint8)

    # Invert the binary mask so that object pixels are set to 1 and background pixels are set to 0
    binary_mask = 1 - binary_mask

    # Convert numpy array to PIL image
    binary_image = Image.fromarray(binary_mask * 255)

    # Save the binary mask as a .png file
    binary_image.save(output_mask_path)


def my_function_star(args):
    return segmentation_mask_to_binary(*args)


def recover_dataset(vid_root, mask_root):
    src_mask_root = mask_root + "_xmem_infer"
    dst_mask_root = mask_root + "_xmem_infer_recover"

    # pre-load the mapping file
    data_info = json.load(open(os.path.join(os.path.dirname(src_mask_root), "info.json")))
    new2old_mapping = data_info['mapping']['new2old']

    vid_list = sorted(os.listdir(vid_root))
    assert vid_list == sorted(os.listdir(mask_root)) == sorted(os.listdir(src_mask_root))

    job_list = []

    for vid in tqdm(vid_list):
        vid_folder = os.path.join(vid_root, vid)
        img_list = sorted(os.listdir(vid_folder))
        reference_image = Image.open(os.path.join(vid_folder, img_list[0]))

        mask_src_vid = os.path.join(src_mask_root, vid)
        mask_dst_vid = os.path.join(dst_mask_root, vid)
        os.makedirs(mask_dst_vid, exist_ok=True)

        exp_list = sorted(os.listdir(mask_src_vid))
        for exp in tqdm(exp_list):
            mask_src_vid_exp = os.path.join(mask_src_vid, exp, "masks")
            mask_dst_vid_exp = os.path.join(mask_dst_vid, exp)
            os.makedirs(mask_dst_vid_exp, exist_ok=True)

            if not os.path.exists(mask_src_vid_exp):
                print(mask_src_vid_exp)
                continue
            mask_list = sorted(os.listdir(mask_src_vid_exp))
            for src_mask in mask_list:
                save_name = os.path.join(mask_dst_vid_exp, new2old_mapping[vid][src_mask.split('.')[0]] + '.png')
                job_list.append((os.path.join(mask_src_vid_exp, src_mask), reference_image.size, save_name))

    print("num jobs: {}".format(len(job_list)))
    num_processes = 64
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(my_function_star, job_list), total=len(job_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video frames given a few (1+) existing annotation masks')
    parser.add_argument('--video', type=str,
                        help='Path to the video file or directory with .jpg video frames to process', required=True)
    parser.add_argument('--masks', type=str,
                        help='Path to the directory with individual .png masks  for corresponding video frames, named `frame_000000.png`, `frame_000123.png`, ... or similarly (the script searches for the first integer value in the filename). '
                             'Will use all masks int the directory.', required=True)
    args = parser.parse_args()

    sanity_check(args.video, args.masks)
    recover_dataset(args.video, args.masks)

