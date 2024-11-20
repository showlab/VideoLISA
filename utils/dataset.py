import glob
import os
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

# image datasets
from .data_processing import get_mask_from_json
from .sem_seg_dataset import SemSegPseudoVidDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegPseudoVidDataset
from .reason_seg_dataset import ReasonSegPseudoVidDataset
from .vqa_dataset import VQADataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN)

# video datasets
from .vos_dataset import YTVOS_VideoLISA_Dataset
from .refer_vos_dataset import ReferYTVOSDataset
from .ref_davis_dataset import RefDAVIS_VideoLISA_Dataset
from .mevis_dataset import MeViS_VideoLISA_Dataset

import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    valid_indices = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        valid_idx,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        valid_indices.append(valid_idx)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        raise ValueError

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()

    conv = conversation_lib.default_conversation.copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    assert conv_type == "phi3_instruct"
    sep = conv.sep + conv.roles[1]

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # ------------- adapted from LLaVA Phi-3 ------------------
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1
            # ------------- end line ------------------

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "valid_indices": valid_indices,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class ImgVidHybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        video_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="refer_seg_video||vid_qa",
        sample_rate=[1, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        reason_seg_data="ReasonSeg|train",
        vqa_data="llava_instruct_150k",
        ref_vos_data="refer_youtube_vos||mevis",
        vos_data="ytvos||mose",
        explanatory=0.1,
        num_frames_sparse=50,
        num_frames_dense=4,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split(",")
        assert len(self.datasets) == len(sample_rate), (len(self.datasets), len(sample_rate), sample_rate)
        sample_rate_expand = []

        self.all_datasets = []
        for idx, dataset in enumerate(self.datasets):
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegPseudoVidDataset(
                        base_image_dir,
                        tokenizer,
                        video_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        num_frames_sparse,
                        num_frames_dense,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegPseudoVidDataset(
                        base_image_dir,
                        tokenizer,
                        video_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        num_frames_sparse,
                        num_frames_dense,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegPseudoVidDataset(
                        base_image_dir,
                        tokenizer,
                        video_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        num_frames_sparse,
                        num_frames_dense,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        video_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        num_frames_sparse,
                        num_frames_dense,
                    )
                )
                sample_rate_expand.append(sample_rate[idx])
            elif dataset == "ref_vos":
                vid_datasets = ref_vos_data.split(",")
                for vid_data in vid_datasets:
                    if vid_data == "refer_youtube_vos":
                        self.all_datasets.append(
                            ReferYTVOSDataset(
                                base_image_dir,
                                tokenizer,
                                video_tower,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_sparse,
                                num_frames_dense,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    elif vid_data == "mevis":
                        self.all_datasets.append(
                            MeViS_VideoLISA_Dataset(
                                base_image_dir,
                                tokenizer,
                                video_tower,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_sparse,
                                num_frames_dense,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    elif vid_data == "davis":
                        self.all_datasets.append(
                            RefDAVIS_VideoLISA_Dataset(
                                base_image_dir,
                                tokenizer,
                                video_tower,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_sparse,
                                num_frames_dense,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    else:
                        raise NotImplementedError
            elif dataset == "vos":
                vid_datasets = vos_data.split(",")
                for vid_data in vid_datasets:
                    if vid_data == "ytvos":
                        self.all_datasets.append(
                            YTVOS_VideoLISA_Dataset(
                                base_image_dir,
                                tokenizer,
                                video_tower,
                                samples_per_epoch,
                                precision,
                                image_size,
                                num_classes_per_sample,
                                exclude_val,
                                num_frames_sparse,
                                num_frames_dense,
                            )
                        )
                        sample_rate_expand.append(sample_rate[idx])
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError

        assert len(self.all_datasets) == len(sample_rate_expand)
        sample_rate = np.array(sample_rate_expand)
        for idx in range(len(sample_rate)):
            print("Dataset: {}, sample rate: {}".format(self.all_datasets[idx], sample_rate[idx]))
        self.sample_rate = sample_rate / sample_rate.sum()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.all_datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class VideoValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        num_frames_sparse=50,
        num_frames_dense=4,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg"))
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        # "images/mscoco/images/train2014",
                        "refer_seg/images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.num_frames_sparse = num_frames_sparse
        self.num_frames_dense = num_frames_dense

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this video? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # Repeat image into video
        image_clip = torch.stack([image_clip] * self.num_frames_sparse, dim=0)
        image = torch.stack([image] * self.num_frames_dense, dim=0)
        masks = torch.cat([masks] * self.num_frames_dense, dim=0)

        assert image_clip.shape[0] == self.num_frames_sparse
        assert image.shape[0] == self.num_frames_dense
        assert masks.shape[0] == self.num_frames_dense or masks.shape[0] == 0, masks.shape

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            list(range(self.num_frames_dense)),
            resize,
            None,
            None,
            inference,
        )


class ReasonSegTestDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        num_frames_sparse=50,
        num_frames_dense=4,
    ):
        # self.base_image_dir = base_image_dir
        self.base_image_dir = "/home/ubuntu/data/ReasonSeg"
        splits = val_dataset.split("|")
        assert len(splits) == 3

        ds, split, query_type = splits
        images = glob.glob(
            os.path.join(self.base_image_dir, split, "*.jpg")
        )

        if query_type == "all":
            images_query_type = images
        elif query_type == "long":
            images_query_type = []
            for image_path in images:
                json_path = image_path.replace(".jpg", ".json")
                try:
                    with open(json_path, "r") as r:
                        anno = json.loads(r.read())
                except:
                    with open(json_path, "r", encoding="cp1252") as r:
                        anno = json.loads(r.read())
                is_sentence = anno["is_sentence"]
                if is_sentence:
                    images_query_type.append(image_path)
        else:
            assert query_type == "short"
            images_query_type = []
            for image_path in images:
                json_path = image_path.replace(".jpg", ".json")
                try:
                    with open(json_path, "r") as r:
                        anno = json.loads(r.read())
                except:
                    with open(json_path, "r", encoding="cp1252") as r:
                        anno = json.loads(r.read())
                is_sentence = anno["is_sentence"]
                if not is_sentence:
                    images_query_type.append(image_path)

        self.images = images_query_type
        self.data_type = "reason_seg"
        self.query_type = query_type

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.num_frames_sparse = num_frames_sparse
        self.num_frames_dense = num_frames_dense

    def __len__(self):
        return len(self.images)

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

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_path = image_path.replace(".jpg", ".json")
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this video? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # Repeat image into video
        image_clip = torch.stack([image_clip] * self.num_frames_sparse, dim=0)
        image = torch.stack([image] * self.num_frames_dense, dim=0)
        masks = torch.cat([masks] * self.num_frames_dense, dim=0)

        assert image_clip.shape[0] == self.num_frames_sparse
        assert image.shape[0] == self.num_frames_dense
        assert masks.shape[0] == self.num_frames_dense or masks.shape[0] == 0, masks.shape

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            list(range(self.num_frames_dense)),
            resize,
            None,
            None,
            inference,
        )

