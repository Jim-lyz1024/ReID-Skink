import os
import re
import json


def filter_by_regex(lst, regex):
    pattern = re.compile(regex)
    return [x for x in lst if pattern.search(x)]

def produce_interleave_json(train_dir, pattern_dir, animal_name):
    """
    Produce interleaved-template JSON annotation file for InterleaveReIDJsonDataset.
    """
    # Check whether the train directory and the pattern directory exist.
    if not os.path.exists(train_dir):
        raise FileExistsError(f"Train directory: {train_dir} does not exist.")
    if not os.path.exists(pattern_dir):
        raise FileExistsError(f"Pattern directory: {pattern_dir} does not exist.")
    
    # Ensure the destination directory exists.
    # os.makedirs(output_json_dir, exist_ok = True)

    data = []
    train_files = sorted(os.listdir(train_dir))
    pattern_files = sorted(os.listdir(pattern_dir))
    for train_file in train_files:
        if not train_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        train_fname, ext = os.path.splitext(train_file)
        aid = int(train_fname.split("_")[0])
        target_image_path = os.path.join(train_dir, train_file)
        safe_train_fname = re.escape(train_fname)
        pattern_images = filter_by_regex(lst = pattern_files, regex = r"{}_p.*".format(safe_train_fname))
        pattern_image_paths = [os.path.join(pattern_dir, p) for p in pattern_images]
        if len(pattern_image_paths) == 0:
            print(pattern_images)
            print(f"[Warning] No pattern images found for {train_file}.\n")
            continue
        item = {
            "text_template": f"A photo of a {animal_name} individual with identity {str(aid)}. This individual has <img0> <img1> <img2> <img3> <img4> patterns.", 
            "pattern_image_paths": pattern_image_paths, 
            "target_image_path": target_image_path, 
            "identity": aid
            }
        # print(item)
        # print()
        data.append(item)
    return data


# ====== CONFIG ======
# ROOT_DIR = "/data/yil708/Code-Skink/sam3/dataset"
# TRAIN_DIR = os.path.join(ROOT_DIR, "train")
# PATTERN_DIR = os.path.join(ROOT_DIR, "pattern")
ROOT_DIR = "/data/yil708/Code-Skink/sam3/dataset/reid_skink_data/try/Skink"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
PATTERN_DIR = os.path.join(ROOT_DIR, "pattern")

OUTPUT_JSON = "skink_interleave_prompts.json"
ANIMAL_NAME = "skink"
# ====================

data = produce_interleave_json(train_dir = TRAIN_DIR, 
                               pattern_dir = PATTERN_DIR, 
                               animal_name = ANIMAL_NAME)
# Write JSON file
with open(OUTPUT_JSON, "w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2, ensure_ascii = False)

print(f"Saved {len(data)} items to {OUTPUT_JSON}")
