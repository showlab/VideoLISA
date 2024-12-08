import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.VideoLISA import VideoLISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.video_capture import VideoCapture


def parse_args(args):
    parser = argparse.ArgumentParser(description="VideoLISA Inference")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument("--vis_save_path", default="vis", type=str)
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--num_frames_sparse", default=50, type=int)
    parser.add_argument("--num_frames_dense", default=4, type=int)
    parser.add_argument(
        "--conv_type",
        default="phi3_instruct",
        type=str,
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


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


def get_dense_indices(num_frames_temporal, num_frames_dense):
    intervals = np.linspace(start=0, stop=num_frames_temporal - 1, num=num_frames_dense + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def main(args):
    # ---------------------------- config env ------------------------------------
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # ---------------------------- prepare model ------------------------------------
    model = VideoLISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    else:
        raise NotImplementedError

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    while True:
        # Prepare text prompt
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "Sure, [SEG].")
        prompt = conv.get_prompt()

        # Prepare video
        video_path = input("Please input the video path: ")
        if not os.path.exists(video_path):
            print("File not found in {}".format(video_path))
            raise FileNotFoundError
        vid_save_dir = os.path.join(args.vis_save_path, os.path.basename(video_path).split('.')[0])
        os.makedirs(vid_save_dir, exist_ok=True)

        frame_list = VideoCapture.load_all_frames_from_video(video_path)
        total_frames = len(frame_list)
        print("Video loaded, preparing input.")

        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_sparse)
        valid_dense_idxs = get_dense_indices(args.num_frames_sparse, args.num_frames_dense)

        # pre-process images
        image_list_sam, image_list_clip, image_list_np = [], [], []

        for frm_idx in sparse_idxs:
            image_np = frame_list[frm_idx]
            original_size_list = [image_np.shape[:2]]

            image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            image_list_clip.append(image_clip)

        for frm_idx in range(total_frames):
            image_np = frame_list[frm_idx]
            original_size_list = [image_np.shape[:2]]

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            image_list_sam.append(image)
            image_list_np.append(image_np)

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # stack as video
        image = torch.stack(image_list_sam, dim=1)
        image_clip = torch.stack(image_list_clip, dim=1)

        print("Running model inference")
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            dense_indices=[valid_dense_idxs],
            num_frames_sparse=args.num_frames_sparse,
            num_frames_dense=args.num_frames_dense,
        )

        print("Inference done. Saving results...\n")
        for i, pred_mask_vid in enumerate(pred_masks):
            if pred_mask_vid.shape[0] == 0:
                continue

            assert total_frames == pred_mask_vid.shape[0]

            for frame_idx in range(total_frames):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0

                save_path = "{}/img_{:03d}.png".format(vid_save_dir, frame_idx)
                binary_mask = np.where(pred_mask > 0, 1, 0)
                cv2.imwrite(save_path, binary_mask * 255)

                if args.save_overlay:
                    save_path = "{}/masked_img_{:03d}.jpg".format(vid_save_dir, frame_idx)
                    save_img = image_list_np[frame_idx].copy()
                    save_img[pred_mask] = (
                            image_list_np[frame_idx] * 0.5
                            + pred_mask[:, :, None].astype(np.uint8) * np.array([0, 0, 255]) * 0.5
                    )[pred_mask]
                    cv2.imwrite(save_path, save_img)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
