import os
import numpy as np
import torch
from PIL import Image
import cv2

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam3Processor,
    Sam3Model,
)


image_folder = "./Segmentation/Val/Image"
mask_folder  = "./Segmentation/Val/masks_correct"

ground_ckpt = "IDEA-Research/grounding-dino-base"

PROMPTS = [
    "person",
    "human body",
    "decomposed body",
    "body placement",
    "upper body",
    "lower body",
    "torso",
    "half body",
    "head",
    "hand",
    "arm",
    "leg",
    "foot",
    "human remains",
    "body part",

]

BOX_THRESH  = 0.15
TEXT_THRESH = 0.15
MIN_BOX_AREA_FRAC = 0.10


sam3_ckpt = "facebook/sam3"


SAVE_EXT = ".png"


USE_HOLE_FILL = True


USE_MORPH_CLOSE = False
CLOSE_KERNEL_SIZE = 3
CLOSE_ITERS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(mask_folder, exist_ok=True)


valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


gd_processor = AutoProcessor.from_pretrained(ground_ckpt)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(ground_ckpt).to(device).eval()

sam_processor = Sam3Processor.from_pretrained(sam3_ckpt)
sam3 = Sam3Model.from_pretrained(sam3_ckpt).to(device).eval()


def postprocess_grounding(outputs, input_ids, target_sizes, box_thresh, text_thresh):
    try:
        return gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        return gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=target_sizes,
        )[0]


def pick_best_box_multi_prompt(image_pil):
    W, H = image_pil.size
    img_area = float(W * H)

    best_score = -1.0
    best_box = None
    best_prompt = None

    for prompt in PROMPTS:
        inputs = gd_processor(images=image_pil, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = gd_model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=device)
        results = postprocess_grounding(outputs, inputs.input_ids, target_sizes, BOX_THRESH, TEXT_THRESH)

        if "boxes" not in results or len(results["boxes"]) == 0:
            continue

        boxes = results["boxes"].detach().float().cpu().numpy()
        scores = results["scores"].detach().float().cpu().numpy()

        if MIN_BOX_AREA_FRAC > 0:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            keep = areas >= (MIN_BOX_AREA_FRAC * img_area)
            if not np.any(keep):
                continue
            boxes, scores = boxes[keep], scores[keep]

        i = int(np.argmax(scores))
        if float(scores[i]) > best_score:
            best_score = float(scores[i])
            best_box = boxes[i].tolist()
            best_prompt = prompt

    return best_box, best_score, best_prompt


def apply_morphological_closing(mask_bin):
    """
    Tiny optional closing only.
    Use carefully because large closing can spill into background.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(
        mask_bin,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=CLOSE_ITERS
    )
    return (closed > 0).astype(np.uint8) * 255


def fill_internal_holes(mask_bin):
    """
    Fill only black holes fully enclosed inside white foreground.
    Safe even when foreground touches image borders.
    """
    mask_bin = (mask_bin > 0).astype(np.uint8) * 255

    padded = cv2.copyMakeBorder(
        mask_bin,
        1, 1, 1, 1,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    flood = padded.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, flood_inv)
    filled = filled[1:-1, 1:-1]

    return (filled > 0).astype(np.uint8) * 255


def sam3_mask_from_box(image_pil, box_xyxy, stem=None):
    sam_inputs = sam_processor(
        images=image_pil,
        input_boxes=[[box_xyxy]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        sam_out = sam3(**sam_inputs)

    pp = sam_processor.post_process_instance_segmentation(
        sam_out,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=sam_inputs["original_sizes"].tolist(),
    )[0]

    masks = pp.get("masks", [])
    if len(masks) == 0:
        return None

    if "scores" in pp and len(pp["scores"]) == len(masks):
        m_i = int(torch.tensor(pp["scores"]).argmax().item())
    else:
        m_i = 0

    mask = masks[m_i]
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    raw_mask_bin = (mask > 0).astype(np.uint8) * 255

    if USE_HOLE_FILL:
        filled_mask_bin = fill_internal_holes(raw_mask_bin)
    else:
        filled_mask_bin = raw_mask_bin.copy()

    if USE_MORPH_CLOSE:
        closed_mask_bin = apply_morphological_closing(filled_mask_bin)
    else:
        closed_mask_bin = filled_mask_bin.copy()

    final_mask_bin = (closed_mask_bin > 0).astype(np.uint8) * 255

    return final_mask_bin



image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)]
image_files.sort()

print("Found images:", len(image_files))


failed_det = 0
failed_sam = 0
saved = 0

for fn in image_files:
    img_path = os.path.join(image_folder, fn)
    stem = os.path.splitext(fn)[0]

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[SKIP] {fn} (cannot open): {e}")
        continue

    box, score, used_prompt = pick_best_box_multi_prompt(image)
    if box is None:
        failed_det += 1
        print(f"[NO BOX] {fn} (no detection from prompts)")
        continue

    mask_bin = sam3_mask_from_box(image, box, stem=stem)
    if mask_bin is None:
        failed_sam += 1
        print(f"[NO MASK] {fn} (SAM3 returned no mask) | prompt='{used_prompt}' score={score:.3f}")
        continue

    out_path = os.path.join(mask_folder, stem + SAVE_EXT)
    ok = cv2.imwrite(out_path, mask_bin)

    if not ok:
        print(f"[WRITE FAIL] {fn} -> {out_path}")
        continue

    saved += 1
    print(f"[OK] {fn} -> {out_path} | prompt='{used_prompt}' score={score:.3f}")

print("\nDone.")
print("Saved:", saved)
print("Failed detection:", failed_det)
print("Failed SAM:", failed_sam)
