"""
Modified from https://github.com/henghuiding/MeViS/blob/main/ReferFormer_dataset.py
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import cv2
import json
import numpy as np
import random
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pycocotools import mask as coco_mask

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class MeViS_VideoLISA_Dataset(torch.utils.data.Dataset):
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
        self.base_image_dir = "/data_sdf/LLM_DATA/video_centric/RefVOS/mevis"
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.num_frames_sparse = num_frames_sparse
        self.num_frames_dense = num_frames_dense

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.MeViS_train = MeViSDataset(img_folder=os.path.join(self.base_image_dir, "train"),
                                        ann_file=os.path.join(self.base_image_dir,
                                                              "train/meta_expressions.json"),
                                        num_frames=self.num_frames_sparse)

        self.MeViS_valid_u = MeViSDataset(img_folder=os.path.join(self.base_image_dir, "valid_u"),
                                          ann_file=os.path.join(self.base_image_dir,
                                                                "valid_u/meta_expressions.json"),
                                          num_frames=self.num_frames_sparse)
        print('Using MeViS dataset, {} from train, {} from valid_u'.format(len(self.MeViS_train), len(self.MeViS_valid_u)))
        self.datasets_trainval = [self.MeViS_train, self.MeViS_valid_u]
        sample_rate = np.array([len(self.MeViS_train), len(self.MeViS_valid_u)])
        self.sample_rate = sample_rate / sample_rate.sum()

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
        ind = np.random.choice(list(range(len(self.datasets_trainval))), p=self.sample_rate)
        dataset_cls = self.datasets_trainval[ind]

        idx = random.randint(0, len(dataset_cls) - 1)
        image_list, target = dataset_cls.__getitem__(idx)

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
        for idx in range(target["masks"].shape[0]):
            if idx in dense_indices:
                mask_list.append(target["masks"][idx])
        masks = torch.stack(mask_list, dim=0)

        questions = []
        answers = []
        sampled_classes = [target["caption"]]
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
            sampled_classes
        )


class MeViSDataset(Dataset):
    """
    A dataset class for the MeViS dataset which was first introduced in the paper:
    "MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions"
    """

    def __init__(self,
                 img_folder,
                 ann_file,
                 num_frames,
                 ):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.subset = os.path.basename(img_folder)
        self.prepare_metas()

        mask_json = os.path.join(str(self.img_folder) + '/mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            cnt = 0
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                    meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['category'] = 0
                    self.metas.append(meta)
                    if self.subset == "valid_u":
                        break
                cnt += 1
                if self.subset == "valid_u" and cnt == 8:
                    break

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        video, exp, anno_id, category, frames, frame_id = \
            meta['video'], meta['exp'], meta['anno_id'], meta['category'], meta['frames'], meta['frame_id']
        # clean up the caption
        exp = " ".join(exp.lower().split())
        category_id = 0
        vid_len = len(frames)

        num_frames = self.num_frames
        # random sparse sample
        sample_indx = [frame_id]
        if self.num_frames != 1:
            # local sample
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    num_repeat = global_n // vid_len
                    select_id = random.sample(range(vid_len), global_n % vid_len) + list(range(vid_len)) * num_repeat
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                    assert len(sample_indx) == self.num_frames

        sample_indx.sort()          # ensure the video in correct temporal order

        # read frames and masks
        imgs, labels, boxes, masks, valid = [], [], [], [], []
        for j in range(self.num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
            img = Image.open(img_path).convert('RGB')
            mask = np.zeros(img.size[::-1], dtype=np.float32)
            for x in anno_id:
                frm_anno = self.mask_dict[x][frame_indx]
                if frm_anno is not None:
                    mask += coco_mask.decode(frm_anno)

            # create the target
            label = torch.tensor(category_id)

            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:  # some frame didn't contain the instance
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)
            mask = torch.from_numpy(mask)

            # use cv2 to read image for consistency with other datasets
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # append
            imgs.append(img)
            labels.append(label)
            masks.append(mask)
            boxes.append(box)

        # transform
        h, w = imgs[0].shape[:2]
        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        target = {
            'frames_idx': torch.tensor(sample_indx),  # [T,]
            'labels': labels,  # [T,]
            'boxes': boxes,  # [T, 4], xyxy
            'masks': masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            'caption': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }

        return imgs, target


if __name__ == '__main__':
    base_image_dir = "/data_sdf/LLM_DATA/video_centric/RefVOS/mevis"

    data_train = MeViSDataset(img_folder=os.path.join(base_image_dir, "train"),
                              ann_file=os.path.join(base_image_dir, "train/meta_expressions.json"),
                              num_frames=32)

    data_valid_u = MeViSDataset(img_folder=os.path.join(base_image_dir, "valid_u"),
                                ann_file=os.path.join(base_image_dir, "valid_u/meta_expressions.json"),
                                num_frames=32)

    print("{} samples from MeViS train".format(len(data_train)))
    print("{} samples from MeViS test".format(len(data_valid_u)))

