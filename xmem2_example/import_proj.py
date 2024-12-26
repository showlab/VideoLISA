import os
import json
import argparse
import shutil
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(".")
from util.image_loader import PaletteConverter


def resize_preserve(img, size, interpolation):
    h, w = img.height, img.width

    # Resize preserving aspect ratio
    new_w = (w * size // min(w, h))
    new_h = (h * size // min(w, h))

    resized_img = img.resize((new_w, new_h), resample=interpolation)

    return resized_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        help='The name of the project to use (name of the corresponding folder in the workspace). Will be created if doesn\'t exist ',
                        required=False,
                        default="RefYTVOS-VideoLISA")
    parser.add_argument('--size', type=int,
                        help='The name of the project to use (name of the corresponding folder in the workspace). Will be created if doesn\'t exist ',
                        default=0)
    parser.add_argument('--videos', type=str, help='Path to the folder with video frames', required=False,
                        default="/home/ubuntu/XMem2/refytvos_custom/JPEGImages")
    parser.add_argument('--masks', type=str, help='Path to the folder with existing masks', required=False,
                        default="/home/ubuntu/XMem2/refytvos_custom/Annotations-VideoLISAPhi")
    args = parser.parse_args()

    p_project = os.path.join("workspace", str(args.name))
    if os.path.exists(p_project):
        print(f"Found the project {args.name} in the workspace.")
    else:
        print(f"Creating new project {args.name} in the workspace.")

    old2new_mapping = defaultdict(dict)
    new2old_mapping = defaultdict(dict)

    print("Processing Images")
    if args.videos is not None:
        vid_path_input = args.videos
        vid_path_output = os.path.join(p_project, "videos")
        os.makedirs(vid_path_output, exist_ok=True)

        vid_list = sorted(os.listdir(vid_path_input))
        for vid_idx in tqdm(range(len(vid_list))):
            img_path_input = os.path.join(vid_path_input, vid_list[vid_idx])
            img_path_output = os.path.join(vid_path_output, vid_list[vid_idx])
            os.makedirs(img_path_output, exist_ok=True)

            img_list = sorted(os.listdir(img_path_input))
            for img_idx in range(len(img_list)):
                img_file = os.path.join(img_path_input, img_list[img_idx])
                output_name = os.path.join(img_path_output,
                                           'frame_{:06d}.'.format(img_idx) + img_list[img_idx].split('.')[-1])

                if args.size == 0:
                    shutil.copyfile(img_file, output_name)
                else:
                    img = Image.open(img_file)
                    resized_img = resize_preserve(img, args.size, Image.Resampling.BILINEAR)
                    resized_img.save(output_name)

                old2new_mapping[vid_list[vid_idx]][img_list[img_idx].split('.')[0]] = 'frame_{:06d}'.format(img_idx)
                new2old_mapping[vid_list[vid_idx]]['frame_{:06d}'.format(img_idx)] = img_list[img_idx].split('.')[0]

    print("Processing Masks")
    if args.masks is not None:
        from util.palette import davis_palette

        palette_converter = PaletteConverter(davis_palette)

        mask_vid_path_input = args.masks
        mask_vid_path_output = os.path.join(p_project, "masks")
        os.makedirs(mask_vid_path_output, exist_ok=True)

        mask_vid_list = sorted(os.listdir(mask_vid_path_input))
        for vid_idx in tqdm(range(len(mask_vid_list))):
            exp_path_input = os.path.join(mask_vid_path_input, mask_vid_list[vid_idx])
            exp_path_output = os.path.join(mask_vid_path_output, mask_vid_list[vid_idx])
            os.makedirs(exp_path_output, exist_ok=True)

            exp_list = sorted(os.listdir(exp_path_input))
            for exp_idx in range(len(exp_list)):
                mask_path_input = os.path.join(exp_path_input, exp_list[exp_idx])
                mask_path_output = os.path.join(exp_path_output, exp_list[exp_idx])
                os.makedirs(mask_path_output, exist_ok=True)

                mask_list = sorted(os.listdir(mask_path_input))
                for mask_idx in range(len(mask_list)):
                    mask_file = os.path.join(mask_path_input, mask_list[mask_idx])

                    mask = Image.open(mask_file)
                    if args.size == 0:
                        index_mask = palette_converter.image_to_index_mask(mask)
                    else:
                        resized_mask = resize_preserve(mask, args.size, Image.Resampling.NEAREST).convert('P')
                        index_mask = palette_converter.image_to_index_mask(resized_mask)

                    output_name = os.path.join(mask_path_output,
                                               old2new_mapping[mask_vid_list[vid_idx]][
                                                   mask_list[mask_idx].split('.')[0]] + '.' +
                                               mask_list[mask_idx].split('.')[-1])
                    index_mask.save(output_name)

        try:
            with open(os.path.join(p_project, "info.json")) as f:
                data = json.load(f)
        except Exception:
            data = {}

        data['num_objects'] = palette_converter._num_objects
        data['mapping'] = {'old2new': old2new_mapping,
                           'new2old': new2old_mapping}

        with open(os.path.join(p_project, "info.json"), 'wt') as f_out:
            json.dump(data, f_out, indent=4)

