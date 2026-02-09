import cv2
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

from collections import Counter
# from statsmodels.stats.multitest import multipletests


def get_mask_from_black_bg(image, destination_dir, file, threshold = 5, bg_mode = "gray", bg_gray = 127, feather = 0):
    """
    Get foreground masks from an image with a black background.
    """
    fn, ext = os.path.splitext(file)
    foreground_masks = (np.max(image, axis = -1) > threshold).astype(np.uint8)    # Shape: (H, W)

    if bg_mode == "black":
        bg_color = np.array([0, 0, 0], dtype = np.uint8)
    elif bg_mode == "mean":
        fg_pixels = image[foreground_masks > 0]
        if fg_pixels.size == 0:
            bg_color = np.array([bg_gray, bg_gray, bg_gray], dtype = np.uint8)
        else:
            bg_color = np.clip(fg_pixels.mean(axis = 0), 0, 255).astype(np.uint8)
    else:
        bg_color = np.array([bg_gray, bg_gray, bg_gray], dtype = np.uint8)
    # print(f"Background Color: {bg_color}")

    if feather > 0:
        kernel_size = feather if feather % 2 == 1 else feather + 1
        soft_mask = cv2.GaussianBlur(foreground_masks.astype(np.float32), (kernel_size, kernel_size), 0)
        soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., None]

        bg_img = np.full_like(image, fill_value = bg_color)
        rgb_for_score = (
            image.astype(np.float32) * soft_mask + bg_img.astype(np.float32) * (1.0 - soft_mask)
        )
        rgb_for_score = np.clip(rgb_for_score, 0, 255).astype(np.uint8)
        destination_path = os.path.join(destination_dir, f"{fn}_{bg_mode}_feather{feather}{ext}")
    else:
        bg_img = np.full_like(image, fill_value = bg_color)
        foreground_masks_3c = foreground_masks[..., None]    # Shape: (H, W, 1)
        rgb_for_score = np.where(foreground_masks_3c == 1, image, bg_img)
        destination_path = os.path.join(destination_dir, f"{fn}_{bg_mode}{ext}")
    # print(f"RGB for Score Shape: {rgb_for_score.shape}")
    # cv2.imwrite(destination_path, rgb_for_score)
    return foreground_masks, rgb_for_score
    
def sobel_mag(gray_f):
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize = 3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize = 3)
    return cv2.magnitude(gx, gy)

def build_texture_score_map(image, foreground_masks, destination_dir, file, margin_px, blur_ks = 0, show_plot = False):
    """
    Build texture score map using Sobel operator within the foreground masks with a margin.
    """
    # plt.figure(figsize = (15, 8))
    # plt.subplot(2, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(image)
    # plt.axis("off")

    fn, ext = os.path.splitext(file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # print(f"Gray Image Shape: {gray_image.shape}")
    if blur_ks > 1:
        gray_image = cv2.GaussianBlur(gray_image, (blur_ks, blur_ks), 0)
    edge = sobel_mag(gray_image)
    # print(f"Sobel Edge Map Shape: {edge.shape}")
    
    # plt.subplot(2, 3, 2)
    # plt.title("Raw Sobel Edge Map")
    # plt.imshow(edge, cmap = "jet")
    # plt.axis("off")

    dist = cv2.distanceTransform((foreground_masks * 255).astype(np.uint8), cv2.DIST_L2, 5)

    # plt.subplot(2, 3, 3)
    # plt.title("Distance Transform")
    # plt.imshow(dist, cmap = "viridis")
    # plt.axis("off")

    inside = ((dist > float(margin_px)) & (foreground_masks > 0)).astype(np.uint8)
    final_score_map = edge * inside.astype(np.float32)
    # print(f"Final Score Map Shape: {final_score_map.shape}")

    # plt.subplot(2, 3, 4)
    # plt.title(f"Inside Mask with Margin {margin_px}px")
    # plt.imshow(inside, cmap = "gray")
    # plt.axis("off")

    # plt.subplot(2, 3, 5)
    # plt.title("Final Score Map")
    # plt.imshow(final_score_map, cmap = "jet")
    # plt.axis("off")

    # plt.tight_layout()
    # destination_path = os.path.join(destination_dir, f"{fn}_texture_score_map_{margin_px}.png")
    # plt.savefig(destination_path)

    # if show_plot:
    #     plt.show()
    return final_score_map.astype(np.float32), inside.astype(np.uint8), dist.astype(np.float32)

def empirical_p_map(score_map, inside):
    """
    Produce an empirical p-values map given the scores.
    """
    inside_bool = inside.astype(bool)
    vals = score_map[inside_bool].astype(np.float64)
    vals_sorted = np.sort(vals)
    N = vals_sorted.size    # number of candidate points

    score_inside = score_map[inside_bool].astype(np.float64)
    idx = np.searchsorted(vals_sorted, score_inside, side = "left")    # number of elements less than each score
    ge = N - idx
    p_inside = (1.0 + ge) / (N + 1.0)
    p_map = np.ones_like(score_map, dtype = np.float64)
    p_map[inside_bool] = p_inside
    return p_map

def bh_fdr(pvals, q = 0.05):
    """
    Perform the Benjamini-Hochberg (BH) test to control FDR.
    """
    all_p_vals = pvals.ravel()
    m = all_p_vals.size
    order = np.argsort(all_p_vals)
    all_p_vals_sorted = all_p_vals[order]
    # print(f"Sorted p-values: {all_p_vals_sorted}")
    thresholds = (np.arange(1, m + 1) / m) * q
    # print(f"thresholds: {thresholds}")
    passed = all_p_vals_sorted <= thresholds
    if not np.any(passed):
        return np.zeros_like(all_p_vals, dtype = bool).reshape(pvals.shape)
    k = np.max(np.where(passed)[0])
    cutoff = all_p_vals_sorted[k]
    return (pvals <= cutoff)

def local_peaks_from_sig(score_map, sig_map, ksize = 3):
    """
    
    """
    sig_bool = sig_map.astype(bool)
    score_sig = score_map * sig_map.astype(np.float32)
    kernel = np.ones((ksize, ksize), np.uint8)
    dil = cv2.dilate(score_sig, kernel)
    # print(f"dil Shape: {dil.shape}")
    peaks = (score_sig >= (dil - 1e-12)) & sig_bool
    return peaks

def nms_pick_k(cands, radius, k):
    """
    Filter candidates using Non-Maximum Suppression (NMS) based on a given radius and the number of points needed.
    """
    # if len(cands) < k:
    #     raise ValueError(f"Number of available candidates ({len(cands)}) is less than K ({k}).")
    chosen = []
    r2 = radius * radius
    for (x, y, s) in cands:    # sorted desc
        ok = True
        for (cx, cy, cs) in chosen:
            if (x - cx) ** 2 + (y - cy) ** 2 < r2:
                ok = False
                break
        if len(chosen) == k:
            break
        if ok:
            chosen.append((x, y, s))
    return chosen

def mode(num_of_chosens):
    """
    Compute the mode of a list of numbers.
    """
    c = Counter(num_of_chosens)
    max_count = max(c.values())
    ms = sorted([k for k, v in c.items() if v == max_count])
    return ms, max_count

def crop_square(image, target_center_x, target_center_y, size):
    """
    Crop an image to a square with a given target center point and a size.
    """
    h, w = image.shape[:2]
    half = size // 2
    x1 = max(0, target_center_x - half)
    y1 = max(0, target_center_y - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)
    image_patch = image[y1:y2, x1:x2]
    return image_patch, (x1, y1, x2, y2)

def topk_points(score, topk = 2000, min_score = 1e-8):
    """
    Choose top-k points from the score map above a minimum score threshold.
    """
    h, w = score.shape
    flat = score.reshape(-1)    # Shape: (H * W,)

    if topk >= flat.size:
        idxs = np.argsort(flat)[::-1]
    else:
        idxs = np.argpartition(flat, -topk)[-topk:]
        idxs = idxs[np.argsort(flat[idxs])[::-1]]
    
    out = []
    for ind in idxs:
        s = float(flat[ind])
        if s <= min_score:
            break
        y = int(ind // w)
        x = int(ind % w)
        out.append((x, y, s))
    return out

def fg_bbox(foreground_masks):
    """
    Get a bounding box of the foreground areas, where (x1, y1) is the top-left point and (x2, y2) is the bottom-right point.
    """
    ys, xs = np.where(foreground_masks > 0)
    if len(xs) == 0:
        return (0, 0, foreground_masks.shape[1], foreground_masks.shape[0])
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)
    



# src_dir = "/raid/ywu840/Data/Animal_sam3/Stoat"
# src_dir = "/home/ywu840/Pattern-Gen/Tiger"
src_dir = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Lizard"
src_files = sorted(os.listdir(src_dir))
dst_dir = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Lizard/pattern"
# dst_dir = "/home/ywu840/Pattern-Gen/local_outputs_new"
os.makedirs(dst_dir, exist_ok = True)
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
MARGIN_RATIO = 0.01
RADIUS_RATIO = 0.04
PATCH_SCALE = 0.3

n_chosens = []
for file in src_files:
    # animal = file.split("_")[0]
    img_path = os.path.join(src_dir, file)
    print(f"Image: {img_path}")
    image = cv2.imread(filename = img_path)
    print(f"Image Shape: {image.shape}")
    H, W = image.shape[:2]
    foreground_masks, rgb_for_score = get_mask_from_black_bg(image, destination_dir = dst_dir, file = file, bg_mode = "mean", feather = 3)
    margin_px = max(1, int(MARGIN_RATIO * min(H, W)))
    texture_score_map, inside, dist = build_texture_score_map(image = rgb_for_score, 
                                                              foreground_masks = foreground_masks, 
                                                              destination_dir = dst_dir, 
                                                              file = file, 
                                                              margin_px = margin_px, 
                                                              show_plot = False)
    inside_bool = inside.astype(bool)
    p_map = empirical_p_map(score_map = texture_score_map, inside = inside)
    ################
    selected_bool = inside_bool & (p_map <= 0.0005)
    ################
    print(f"Number of selected p-values: {selected_bool.sum()}")
    p_inside = p_map[selected_bool]

    sig_inside = bh_fdr(pvals = p_inside, q = 0.05)
    sig_map = np.zeros((H, W), dtype = np.uint8)
    sig_map[selected_bool] = sig_inside.astype(np.uint8)
    print(f"Number of significant points: {sig_map.sum()}")
    if sig_map.sum() == 0:
        # print(f"Image: {img_path}")
        # print(f"Number of selected p-values: {selected_bool.sum()}\n")
        # raise ValueError("No significant points are chosen.")
        print("Fallback to original scores.")
        # continue
        cands = topk_points(score = texture_score_map, topk = 500)
    else:
        local_peaks = local_peaks_from_sig(score_map = texture_score_map, sig_map = sig_map, ksize = 3)
        print(f"Number of peaks: {local_peaks.sum()}")
        ys, xs = np.where(local_peaks)
        cands = [(int(x), int(y), float(texture_score_map[y, x])) for x, y in zip(xs, ys)]
        cands.sort(key = lambda t: t[-1], reverse = True)

    nms_radius_px = max(3, int(RADIUS_RATIO * min(H, W)))
    chosen_points = nms_pick_k(cands = cands, radius = nms_radius_px, k = len(cands))
    print(f"Number of chosen points: {len(chosen_points)}")
    # if len(chosen_points) < 4:
    #     print("Number of chosen points is insufficient.")
    #     cands = topk_points(score = texture_score_map, topk = 500)
    #     new_chosen_points = nms_pick_k(cands = cands, radius = nms_radius_px, k = 10)
    #     for p in new_chosen_points:
    #         if len(chosen_points) == 4:
    #             break
    #         if p not in chosen_points:
    #             chosen_points.append(p)
                
    n_chosens.append(len(chosen_points))

    # patch_size = int(max(16, round(PATCH_SCALE * min(H, W))))
    # if patch_size % 2 == 1:
    #     patch_size += 1
    # print(f"Patch Size: {patch_size}")
    
    # fg255 = (foreground_masks * 255).astype(np.uint8)
    # rgb_patches, mask_patches, infos = [], [], []
    # plt.figure(figsize = (10, 8))
    # for idx, (cx, cy, score) in enumerate(chosen_points):
    #     rgb_patch, rgb_bbox = crop_square(image = rgb_for_score, target_center_x = cx, target_center_y = cy, size = patch_size)
    #     mask_patch, mask_bbox = crop_square(image = fg255, target_center_x = cx, target_center_y = cy, size = patch_size)
    #     fg_cov = float((mask_patch > 0).mean())

    #     plt.subplot(2, 2, idx + 1)
    #     plt.title(f"Patch {idx + 1} ({fg_cov:.3f})")
    #     plt.imshow(rgb_patch)
    #     plt.axis("off")
        
        # fn, ext = os.path.splitext(file)
        # output_path = os.path.join(dst_dir, f"{fn}_pattern_{idx + 1}{ext}")
        # cv2.imwrite(output_path, rgb_patch)
    print()
    # plt.tight_layout()
    # plt.show()
mode_k, freq = mode(num_of_chosens = n_chosens)
print(f"Modes: {mode_k}, Freq: {freq}")

"""
selected_points = topk_points(score = texture_score_map, topk = TOPK)
nms_radius_px = max(3, int(RADIUS_RATIO * min(H, W)))
kept_candidates = nms_candidates(cands = selected_points, radius = nms_radius_px)
print(f"Number of candidates left: {len(kept_candidates)}")
if len(kept_candidates) < 10:
    raise RuntimeError("Number of candidates is too few. Please relax the MARGIN_RATIO.")

x1, y1, x2, y2 = fg_bbox(foreground_masks)
# cv2.imwrite(os.path.join(dst_dir, "fg_bbox.png"), image[y1:y2, x1:x2, :])
min_dist_px = max(5, int(MIN_DIST_RATIO * min(H, W)))
final_chosen_points = select_k4_by_quadrants(cands = kept_candidates, 
                                                bbox = (x1, y1, x2, y2), 
                                                min_dist_px = min_dist_px)
if len(final_chosen_points) < 4:
    raise RuntimeError("Failed to select 4 centers.")

patch_size = int(max(16, round(PATCH_SCALE * min(H, W))))
if patch_size % 2 == 1:
    patch_size += 1

fg255 = (foreground_masks * 255).astype(np.uint8)
# rgb_patches, mask_patches, infos = [], [], []
# plt.figure(figsize = (10, 8))
for idx, (cx, cy, score) in enumerate(final_chosen_points):
    rgb_patch, rgb_bbox = crop_square(image = rgb_for_score, target_center_x = cx, target_center_y = cy, size = patch_size)
    mask_patch, mask_bbox = crop_square(image = fg255, target_center_x = cx, target_center_y = cy, size = patch_size)
    fg_cov = float((mask_patch > 0).mean())

    # plt.subplot(2, 2, idx + 1)
    # plt.title(f"Patch {idx + 1} ({fg_cov:.3f})")
    # plt.imshow(rgb_patch)
    # plt.axis("off")
    
    fn, ext = os.path.splitext(file)
    output_path = os.path.join(dst_dir, f"{fn}_pattern_{idx + 1}{ext}")
    cv2.imwrite(output_path, rgb_patch)
print()
# plt.tight_layout()
# plt.show()
"""