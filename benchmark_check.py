import os
import json

data_dir = "ReasonVOS"
anno_file = os.path.join(data_dir, "meta_expressions.json")
video_path = os.path.join(data_dir, "JPEGImages")
mask_path = os.path.join(data_dir, "Annotations")

with open(anno_file, "r") as f:
    anno_data = json.load(f)
print("Num of videos: ", len(anno_data['videos']))

cnt_vipseg = 0
cnt_burst = 0
cnt_mose = 0
cnt_mevis = 0

cnt_text_exp = 0

for video_name in anno_data['videos']:

    item = anno_data['videos'][video_name]
    """
    "source": Dataset source,
    "frames": A list of frame ids,
    "expressions": Language expressions and corresponding object ids.
    """

    if item["source"].lower() == "vipseg":
        cnt_vipseg += 1
    elif item["source"].lower() == "burst":
        cnt_burst += 1
    elif item["source"].lower() == "mose":
        cnt_mose += 1
    elif item["source"].lower() == "mevis":
        cnt_mevis += 1
    else:
        raise ValueError

    cnt_text_exp += len(item['expressions'])


print("{} videos from VIPSeg.".format(cnt_vipseg))
print("{} videos from BURST.".format(cnt_burst))
print("{} videos from MOSE.".format(cnt_mose))
print("{} videos from MeViS.".format(cnt_mevis))
print("{} total language expressions.".format(cnt_text_exp))


# check video and mask sanity
for video_name in anno_data['videos']:
    assert os.path.exists(os.path.join(video_path, video_name))
print("All videos exist.")

mask_list = []
for vid_name in anno_data['videos'].keys():
    vid = anno_data['videos'][vid_name]
    src_dataset = vid['source']
    for sample in vid['expressions']:
        obj_id = sample['obj_id']
        exp_id = sample['exp_id']
        exp_text = sample['exp_text']
        is_sent = sample['is_sent']
        vid_key = f"{src_dataset}_{vid_name}_{obj_id}"
        mask_list.append(vid_key)
print("Num of unique masks:", len(set(mask_list)))

for mask_name in mask_list:
    assert os.path.exists(os.path.join(mask_path, mask_name))
print("All masks exist.")

