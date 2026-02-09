import cv2
import matplotlib
import os
import requests
import torch
import numpy as np

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def segment_image(image_path, seg_prompt, processor):
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, IOError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None
    
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state = inference_state, prompt = seg_prompt)
    # print(output)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    binary_masks = masks.squeeze(0).int()
    # print(f"Masks({binary_masks.shape}): {binary_masks}\n")
    # print(f"Boxes: {boxes}\n")
    # print(f"Scores: {scores.cpu().numpy()}\n")
    return binary_masks, boxes, scores

def apply_mask_black_bg(image, mask, destination_dir, file, mode):
    """
    Apply binary masks to the image with a black background.
    """
    # Ensure mask dimensions match image dimensions
    if mask.shape != image.shape[:2]:
        # If mask is transposed, transpose it back
        if mask.shape == (image.shape[1], image.shape[0]):
            mask = mask.T
        else:
            # Resize mask to match image dimensions
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
    masked_image = image * mask_3d
    fn, ext = os.path.splitext(file)
    # destination_path = os.path.join(destination_dir, f"{fn}_{mode}_black{ext}")
    destination_path = os.path.join(destination_dir, file)
    cv2.imwrite(destination_path, masked_image.astype(np.uint8))
    return masked_image.astype(np.uint8)

def apply_mask_transparent_bg(image, mask, destination_dir, file, mode):
    """
    Apply binary masks to the image with a transparent background.
    """
    b, g, r = cv2.split(image)
    alpha = mask.astype(np.uint8) * 255    # alpha = 255 for foreground, 0 for background
    rgba_image = cv2.merge([b, g, r, alpha])
    fn, ext = os.path.splitext(file)
    destination_path = os.path.join(destination_dir, f"{fn}_{mode}_transparent.png")    # must save as .png to keep transparency
    cv2.imwrite(destination_path, rgba_image)
    return rgba_image

def crop_to_content(image, mask, destination_dir, file, mode):
    """
    Apply binary masks to the image and crop to the detected areas.
    """
    y_indices, x_indices = np.where(mask > 0)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1, :]
    fn, ext = os.path.splitext(file)
    destination_path = os.path.join(destination_dir, f"{fn}_{mode}_cropped{ext}")
    cv2.imwrite(destination_path, cropped_image)
    return cropped_image

def segment_image_with_black_bg(image, mask_image_path, destination_dir, file):
    mask_image = cv2.imread(mask_image_path)
    if image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(mask_image, dsize = (image.shape[1], image.shape[0]))
    
    mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(image, image, mask = binary_mask)
    destination_path = os.path.join(destination_dir, file)
    cv2.imwrite(destination_path, result)
    print(f"{file} has been saved.\n")


# src_dir = "/raid/ywu840/Data/AnimalReID/Stoat/train"
src_dir = "/data/yil708/Code-Skink/sam3/dataset/train"
src_files = sorted(os.listdir(src_dir))
# dst_dir = "/home/ywu840/sam3/outputs"
dst_dir = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Animal"
os.makedirs(dst_dir, exist_ok = True)
MODE = "best"
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
ckpt = "facebook/sam3"
# model = AutoModel.from_pretrained(ckpt).to(DEVICE)
# processor = AutoImageProcessor.from_pretrained(ckpt)
model = build_sam3_image_model()
model = model.to(DEVICE)
processor = Sam3Processor(model, device = DEVICE)
num_of_success = 0
num_of_skipped = 0
num_of_corrupted = 0
corrupted_files = []
# src_files = ["43_-1_0_134_leftphoto.jpg", "44_-1_0_135_leftphoto.jpg", "44_-1_1_135_rightphoto.jpg", "5_-1_13_837_rightphoto.jpg", "69_-1_13_836_rightphoto.jpg"]
for file in src_files:
    fn, ext = os.path.splitext(file)
    
    # Skip if already processed
    output_path = os.path.join(dst_dir, file)
    if os.path.exists(output_path):
        print(f"Skipping {file} - already processed\n")
        num_of_skipped += 1
        continue
    ##########################################
    animal = "Animal"
    # animal = "Lizard"
    ##########################################

    # animal = file.split("_")[0]
    # if animal != "Nyala":
    #     continue
    # if animal == "Nyala":
    #     animal = "Deer"
    # if f"{fn}.png" not in os.listdir("/raid/ywu840/Data/Animal_sam3/temp"):
    #     continue
    img_path = os.path.join(src_dir, file)
    print(f"Image: {img_path}")
    image = cv2.imread(filename = img_path)
    # mask_path = os.path.join("/raid/ywu840/Data/Animal_sam3/temp", f"{fn}.png")
    # segment_image_with_black_bg(image = image, mask_image_path = mask_path, destination_dir = dst_dir, file = file)
    # num_of_success += 1
    print(f"Image Shape: {image.shape}")
    binary_masks, boxes, scores = segment_image(image_path = img_path, seg_prompt = animal, processor = processor)
    
    # Check if image is corrupted
    if binary_masks is None:
        print(f"Skipping corrupted image: {file}\n")
        num_of_corrupted += 1
        corrupted_files.append(file)
        continue
    
    scores = scores.cpu().numpy()
    if len(scores) == 0:
        print(f"No masks detected for {img_path}!\n")
        continue
    elif len(scores) > 1:
        max_score_idx = np.argmax(scores)
        binary_masks = binary_masks.squeeze(1).cpu().numpy()
    else:
        max_score_idx = np.argmax(scores)
        binary_masks = binary_masks.cpu().numpy()
    
    if MODE == "best":
        apply_mask_black_bg(image = image, mask = binary_masks[max_score_idx], destination_dir = dst_dir, file = file, mode = MODE)
        num_of_success += 1
        # apply_mask_transparent_bg(image = image, mask = binary_masks[max_score_idx], destination_dir = dst_dir, file = file, mode = MODE)
        # crop_to_content(image = image, mask = binary_masks[max_score_idx], destination_dir = dst_dir, file = file, mode = MODE)

print(f"\n{'='*60}")
print(f"PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Newly processed files: {num_of_success}")
print(f"Skipped (already exists): {num_of_skipped}")
print(f"Corrupted/unreadable files: {num_of_corrupted}")
print(f"Total processed: {num_of_success + num_of_skipped}")
print(f"{'='*60}")

# Save corrupted files list if any
if corrupted_files:
    corrupted_log = os.path.join(dst_dir, "corrupted_files.txt")
    with open(corrupted_log, "w") as f:
        for corrupted_file in corrupted_files:
            f.write(f"{corrupted_file}\n")
    print(f"\nCorrupted files list saved to: {corrupted_log}")
    print(f"First few corrupted files: {corrupted_files[:5]}")
    if len(corrupted_files) > 5:
        print(f"... and {len(corrupted_files) - 5} more")
# binary_masks, boxes, scores = segment_image(image_path = img_path, seg_prompt = "Sea star", processor = processor)
# binary_masks = binary_masks.squeeze(0).cpu().numpy()
# apply_mask_black_bg(image = image, mask = binary_masks, destination_dir = dst_dir, file = "0_-1_1_3213.jpg")
