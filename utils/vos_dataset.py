'''
References:
https://github.com/Vujas-Eteph/READMem/blob/main/dataset/vos_dataset.py
https://github.com/ttt-matching-based-vos/ttt_matching_vos/blob/main/STCN/dataset/vos_dataset.py
'''
import json
import os
from os import path
import cv2

import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F

from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class YTVOS_VideoLISA_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        num_frames_sparse=50,
        num_frames_dense=4,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        # self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.num_frames_sparse = num_frames_sparse
        self.num_frames_dense = num_frames_dense

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.VOS = VOSDataset(im_root="/data_sdf/LLM_DATA/video_centric/YTVOS/train/JPEGImages",
                              gt_root="/data_sdf/LLM_DATA/video_centric/YTVOS/train/Annotations",
                              max_jump=20,
                              num_frames=num_frames_sparse)
        print("Using YTVOS data.")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_dense_indices(self):
        sequence = np.arange(self.num_frames_sparse)
        random_numbers = np.random.choice(sequence, size=self.num_frames_dense, replace=False)

        return sorted(random_numbers.tolist())

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.VOS) - 1)
        image_list, target_masks, class_name = self.VOS.__getitem__(idx)

        dense_indices = self.get_dense_indices()

        # pre-process for CLIP
        image_clip_list = []
        for img in image_list:
            image_clip = self.clip_image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            image_clip_list.append(image_clip)
        video_data_clip_tsr = torch.stack(image_clip_list, dim=0)

        # pre-process for SAM
        image_sam_list = []
        for idx, image in enumerate(image_list):
            if idx in dense_indices:
                image_sam = self.transform.apply_image(image)
                image_sam_list.append(image_sam)
        resize = image_sam_list[0].shape[:2]

        mask_list = []
        for idx in range(target_masks.shape[0]):
            if idx in dense_indices:
                mask_list.append(torch.from_numpy(target_masks[idx]))
        masks = torch.stack(mask_list, dim=0)

        questions = []
        answers = []
        sampled_classes = [class_name]
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image_sam_list_proc = []
        for image in image_sam_list:
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            image_sam_list_proc.append(image)
        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)

        masks = torch.Tensor(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        assert video_data_clip_tsr.shape[0] == self.num_frames_sparse
        assert image_sam_tsr.shape[0] == masks.shape[0] == self.num_frames_dense

        return (
            None,
            image_sam_tsr,
            video_data_clip_tsr,
            conversations,
            masks,
            label,
            dense_indices,
            resize,
            questions,
            sampled_classes,
        )


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(self, im_root, gt_root, max_jump, num_frames=3, is_bl=False, subset=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        self.meta = json.load(open("/data_sdf/LLM_DATA/video_centric/YTVOS/train/meta.json", "r"))

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

    def __getitem__(self, idx):
        global_trail = 0
        while global_trail < 5:
            if global_trail != 0:
                idx = random.randint(0, len(self.videos))

            video = self.videos[idx]
            info = {}
            info['name'] = video

            vid_im_path = path.join(self.im_root, video)
            vid_gt_path = path.join(self.gt_root, video)
            frames = self.frames[video]

            trials = 0
            while trials < 5:
                info['frames'] = []  # Appended with actual frames

                # ------------------ ReferVOS-like sampling ------------------
                num_frames = self.num_frames
                length = len(frames)
                frame_id = np.random.randint(length)
                frames_idx = [frame_id]
                if self.num_frames != 1:
                    # local sample
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(length - 1, frame_id + sample_id_after)]
                    frames_idx.extend(local_indx)

                    # global sampling
                    if num_frames > 3:
                        all_inds = list(range(length))
                        global_inds = all_inds[:min(frames_idx)] + all_inds[max(frames_idx):]
                        global_n = num_frames - len(frames_idx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id:
                                frames_idx.append(global_inds[s_id])
                        elif length >= global_n:  # sample long range global frames
                            select_id = random.sample(range(length), global_n)
                            for s_id in select_id:
                                frames_idx.append(all_inds[s_id])
                        else:
                            num_repeat = global_n // length
                            select_id = random.sample(range(length), global_n % length) + list(range(length)) * num_repeat
                            # select_id = random.sample(range(length), global_n - length) + list(range(length))
                            for s_id in select_id:
                                frames_idx.append(all_inds[s_id])
                            assert len(frames_idx) == self.num_frames
                frames_idx.sort()  # ensure the video in correct temporal order

                images = []
                masks = []
                for f_idx in frames_idx:
                    jpg_name = frames[f_idx][:-4] + '.jpg'
                    png_name = frames[f_idx][:-4] + '.png'
                    info['frames'].append(jpg_name)

                    this_im = cv2.imread(path.join(vid_im_path, jpg_name))
                    this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)

                    this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                    this_gt = np.array(this_gt)

                    images.append(this_im)
                    masks.append(this_gt)

                labels = np.unique(masks[0])
                # Remove background
                labels = labels[labels != 0]

                if self.is_bl:
                    # Find large enough labels
                    good_lables = []
                    for l in labels:
                        pixel_sum = (masks[0] == l).sum()
                        if pixel_sum > 10 * 10:
                            # OK if the object is always this small
                            # Not OK if it is actually much bigger
                            if pixel_sum > 30 * 30:
                                good_lables.append(l)
                            elif max((masks[1] == l).sum(), (masks[2] == l).sum()) < 20 * 20:
                                good_lables.append(l)
                    labels = np.array(good_lables, dtype=np.uint8)

                if len(labels) == 0:
                    target_object_list = []  # all black if no objects
                    trials += 1
                else:
                    class_list = []
                    for lb in labels:
                        class_list.append(self.meta["videos"][video]["objects"][str(lb)]["category"])

                    sampled_class = np.random.choice(list(set(class_list)), size=1)[0]
                    target_object_list = []
                    for c, lb in zip(class_list, labels):
                        if c == sampled_class:
                            target_object_list.append(lb)
                    break

            if len(target_object_list) == 0:
                assert trials == 5
                global_trail += 1
            else:
                masks = np.stack(masks, 0)
                tar_mask = np.zeros_like(masks).astype(np.float32)
                for tar_obj in target_object_list:
                    assert tar_obj > 0
                    tar_mask += (masks == tar_obj).astype(np.float32)
                break

        return images, tar_mask, sampled_class

    def __len__(self):
        return len(self.videos)

