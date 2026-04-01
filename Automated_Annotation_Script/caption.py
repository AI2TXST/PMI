import json
import re
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image

import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


MODEL_DIR = "./local_models/Qwen2.5-VL-7B-Instruct"

IMAGE_DIR = Path("./PMI_Class_Split")

OUTPUT_JSONL = Path("./multiclass_captions_full.jsonl")
OUTPUT_JSON = Path("./multiclass_captions_full_pretty.json")

BASE_PREFIX = "PMI"
BASE_DIR = IMAGE_DIR.parent

CLASS_NAMES = ["FRESH", "BLOAT", "ACTIVE_DECAY", "ADVANCED_DECAY"]

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MAX_NEW_TOKENS = 180
TEMPERATURE = 0.0
DO_SAMPLE = False

DEVICE_MAP = "auto"
TORCH_DTYPE = "auto"

MAX_RETRIES = 3


def sanitize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


def to_relative_pmi_path(path_obj: Path) -> str:
    return f"{BASE_PREFIX}/{path_obj.relative_to(BASE_DIR).as_posix()}"


def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def enforce_prefix_once(caption: str) -> str:
    caption = sanitize_text(caption)

    caption = re.sub(
        r"^(in this sample,\s*)+",
        "In this sample, ",
        caption,
        flags=re.IGNORECASE
    )

    if not caption.lower().startswith("in this sample,"):
        if caption:
            caption = f"In this sample, {caption[0].lower() + caption[1:]}"
        else:
            caption = "In this sample, a human subject is visible."

    return sanitize_text(caption)


def contains_forbidden_metadata_words(text: str) -> bool:
    t = text.lower()
    forbidden = [
        "postmortem interval", "pmi", "placement day", "season",
        "spring", "summer", "fall", "autumn", "winter", "weather"
    ]
    return any(w in t for w in forbidden)


def contains_forbidden_label_words(text: str) -> bool:
    t = text.lower()
    forbidden = [
        "fresh",
        "bloat",
        "active_decay",
        "advanced_decay",
        "active decay",
        "advanced decay",
        "decomposition stage",
        "early decomposition",
        "late decomposition"
    ]
    return any(w in t for w in forbidden)


def validate_visual_caption(text: str) -> bool:
    t = text.lower()

    if not t.startswith("in this sample,"):
        return False

    if "human" not in t:
        return False

    if contains_forbidden_metadata_words(text):
        return False

    if contains_forbidden_label_words(text):
        return False

    n_sent = count_sentences(text)
    if n_sent < 2 or n_sent > 3:
        return False

    return True


def build_visual_prompt() -> str:
    return """
Identify the subject in the image as a human and provide a formal, scientific visual description based on these strict requirements:

### CORE STRUCTURE
1. Start the caption exactly with the phrase: "In this sample,"
2. Limit the total length to exactly 2 or 3 sentences.
3. Use only clear, formal, and image-grounded language.

### DESCRIPTIVE GUIDELINES
- **Physical State:** Describe posture, orientation, and body completeness. Note if the contour is near normal, swollen, distended, collapsed, leathery, or liquefied.
- **Surface Details:** Note skin condition (peeling, marbling, green discoloration) and presence of insects, maggot activity, or body fluids.
- **Visibility:** Explicitly state if the full subject is not visible, if only parts are visible, or if specific areas are obscured/out of frame.

### PROHIBITED (DO NOT INCLUDE)
- **No Inference:** Do not mention PMI, sex, weather, season, placement day, or decomposition stages (e.g., BLOAT, ACTIVE_DECAY).
- **No Interpretation:** Do not use words like "suggests," "indicates," "likely," or "consistent with." Do not diagnose or state conclusions.
- **No Invisible Features:** If it is not directly visible in the pixels, do not mention it.

### EXAMPLE OF PROPER FORMAT
"In this sample, a human subject is partially visible in a supine orientation, with the lower extremities extending out of the frame. The skin of the exposed torso exhibits green discoloration and marbling, while the abdominal contour appears moderately distended."
""".strip()


def generate_caption(model, processor, image: Image.Image, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE
        )

    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True
    )[0]

    return sanitize_text(output_text)


def generate_visual_with_retry(model, processor, image: Image.Image, max_retry: int = MAX_RETRIES) -> str:
    base_prompt = build_visual_prompt()
    last_caption = ""

    for attempt in range(max_retry):
        if attempt == 0:
            current_prompt = base_prompt
        else:
            current_prompt = (
                base_prompt
                + "\nRewrite the caption and follow all rules exactly."
                + "\nKeep it purely visual, formal, and image-grounded."
                + "\nDo not guess, infer, or imply any class."
            )

        caption = generate_caption(model, processor, image, current_prompt)
        caption = enforce_prefix_once(caption)
        caption = sanitize_text(caption)

        if caption and caption[-1] not in ".!?":
            caption += "."

        last_caption = caption

        if validate_visual_caption(caption):
            return caption

    return last_caption


def gather_images_by_class(
    image_root: Path,
    class_names: List[str],
    allowed_exts: set
) -> List[Dict[str, Any]]:
    selected_records: List[Dict[str, Any]] = []

    for class_name in class_names:
        class_dir = image_root / class_name

        if not class_dir.exists():
            print(f"WARNING: Class folder not found: {class_dir}")
            continue

        class_images = sorted([
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed_exts
        ])

        print(f"{class_name}: found {len(class_images)} images")

        if len(class_images) == 0:
            continue

        for img_path in class_images:
            selected_records.append({
                "class_name": class_name,
                "image_path": img_path
            })

        print(f"{class_name}: selected all {len(class_images)} images")

    return selected_records


print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE_MAP,
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


selected_items = gather_images_by_class(
    image_root=IMAGE_DIR,
    class_names=CLASS_NAMES,
    allowed_exts=ALLOWED_IMAGE_EXTS
)

print(f"\nTotal selected images: {len(selected_items)}")

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

written = 0
errors = 0
records: List[Dict[str, Any]] = []


with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(selected_items, 1):
        image_path = item["image_path"]
        class_name = item["class_name"]

        try:
            image = Image.open(image_path).convert("RGB")

            caption = generate_visual_with_retry(
                model=model,
                processor=processor,
                image=image,
                max_retry=MAX_RETRIES
            )

            record: Dict[str, Any] = {
                "image": to_relative_pmi_path(image_path),
                "label": class_name,
                "caption": caption
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            written += 1

            print(f"[{idx}/{len(selected_items)}] OK: {class_name} | {image_path.name}")
            print(f"  Caption: {caption}")

        except Exception as e:
            errors += 1
            print(f"[{idx}/{len(selected_items)}] ERROR: {class_name} | {image_path.name} -> {e}")


with open(OUTPUT_JSON, "w", encoding="utf-8") as fjson:
    json.dump(records, fjson, ensure_ascii=False, indent=2)

print("\nDone")
print(f"Written: {written}")
print(f"Errors: {errors}")
print(f"Saved JSONL to: {OUTPUT_JSONL}")
print(f"Saved pretty JSON to: {OUTPUT_JSON}")
