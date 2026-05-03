import os
import json
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import evaluate
from aac_metrics import evaluate as aac_evaluate

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
    get_linear_schedule_with_warmup,
)

TRAIN_JSON = "./Correction_Caption/train_captions.json"
VAL_JSON   = "./Correction_Caption/val_captions.json"
TEST_JSON  = "./Correction_Caption/test_captions.json"

ROOT_DIR = ""

MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
BERTSCORE_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"

OUTPUT_DIR = "./blip_caption_correction_outputv6"

MAX_TEXT_LEN = 200
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
NUM_EPOCHS = 30
LR = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
WARMUP_RATIO = 0.1
SEED = 42

GEN_MAX_LENGTH = 200
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 2
LENGTH_PENALTY = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# checkpoint selection: ONLY val_cider_d
PRIMARY_VAL_METRIC = "val_cider_d"
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-4

# report metries
MAIN_METRICS = ["bleu_4", "rouge_l", "meteor", "cider_d", "spider"]
EXTENDED_METRICS = ["bleu_1", "bleu_2", "bleu_3", "bertscore_f1", "clipscore"]
ALL_REPORTED_METRICS = MAIN_METRICS + EXTENDED_METRICS


# UTILS
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_float(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return [float(v) for v in x.detach().cpu().flatten()]
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def normalize_text(s):
    return " ".join(s.strip().lower().split())


def exact_match_score(preds, refs):
    return sum(normalize_text(p) == normalize_text(r) for p, r in zip(preds, refs)) / max(len(preds), 1)


def normalize_metric_key(k: str) -> str:
    k = str(k).strip().lower()
    k = k.replace("-", "_").replace(" ", "_").replace("/", "_")
    if k == "rougel":
        return "rouge_l"
    if k == "cider":
        return "cider_d"
    return k

set_seed(SEED)


# MODEL
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# DATASET
class CaptionDataset(Dataset):
    def __init__(self, json_path, processor, root_dir="", max_text_len=80):
        self.samples = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.processor = processor
        self.root_dir = root_dir
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.samples)

    def resolve_path(self, image_path):
        p = Path(image_path)
        if p.is_absolute():
            return p
        if self.root_dir:
            return Path(self.root_dir) / p
        return p

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = self.resolve_path(item["image"])
        caption = item["caption"].strip()

        image = Image.open(image_path).convert("RGB")
        enc = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        pixel_values = enc["pixel_values"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption,
            "image_path": str(image_path),
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "captions": [x["caption"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
    }


train_dataset = CaptionDataset(TRAIN_JSON, processor, ROOT_DIR, MAX_TEXT_LEN)
val_dataset = CaptionDataset(VAL_JSON, processor, ROOT_DIR, MAX_TEXT_LEN)
test_dataset = CaptionDataset(TEST_JSON, processor, ROOT_DIR, MAX_TEXT_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# METRICS
def load_text_metrics():
    metrics = {}

    metrics["bleu"] = evaluate.load("bleu")
    metrics["rouge"] = evaluate.load("rouge")
    metrics["meteor"] = evaluate.load("meteor")
    metrics["bertscore"] = evaluate.load("bertscore")

    metrics["aac_evaluate"] = aac_evaluate

    # fixed CLIPScore backend
    metrics["clip_processor"] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    metrics["clip_model"] = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    metrics["clip_model"].eval()

    return metrics


TEXT_METRICS = load_text_metrics()

def compute_clipscore(image_paths, preds, batch_size=16):
    clip_model = TEXT_METRICS["clip_model"]
    clip_processor = TEXT_METRICS["clip_processor"]

    all_scores = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_texts = preds[start:start + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = clip_processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            cosine_sim = (image_features * text_features).sum(dim=-1)
            batch_scores = torch.clamp(cosine_sim, min=0.0) * 100.0
            all_scores.extend(batch_scores.detach().cpu().tolist())

    return float(np.mean(all_scores)) if all_scores else 0.0


def compute_overlap_and_semantic_metrics(preds, refs, image_paths):
    """
    Compute everything, then filter only at reporting time.
    """
    results = {}

    # BLEU 
    bleu = TEXT_METRICS["bleu"].compute(
        predictions=preds,
        references=[[r] for r in refs]
    )
    for key, value in bleu.items():
        results[normalize_metric_key(key)] = safe_float(value)

    # ROUGE 
    rouge = TEXT_METRICS["rouge"].compute(
        predictions=preds,
        references=refs
    )
    for key, value in rouge.items():
        results[normalize_metric_key(key)] = safe_float(value)

    # METEOR
    meteor = TEXT_METRICS["meteor"].compute(
        predictions=preds,
        references=refs
    )
    results["meteor"] = safe_float(meteor["meteor"])

    # BERTScore
    bertscore = TEXT_METRICS["bertscore"].compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type=BERTSCORE_MODEL_TYPE,
    )
    results["bertscore_precision"] = float(np.mean(bertscore["precision"]))
    results["bertscore_recall"] = float(np.mean(bertscore["recall"]))
    results["bertscore_f1"] = float(np.mean(bertscore["f1"]))

    # AAC metrics 
    mult_references = [[r] for r in refs]
    corpus_scores, _ = TEXT_METRICS["aac_evaluate"](preds, mult_references)
    for key, value in corpus_scores.items():
        nk = normalize_metric_key(key)
        results[nk] = safe_float(value)

    # CLIPScore fixed implementation
    results["clipscore"] = compute_clipscore(image_paths, preds)

    if "rougel" in results and "rouge_l" not in results:
        results["rouge_l"] = results["rougel"]

    # force required metrics to exist
    missing = [k for k in ALL_REPORTED_METRICS if k not in results]
    if missing:
        raise RuntimeError(f"Missing required reported metrics: {missing}. Available keys: {sorted(results.keys())}")

    return results

# GENERATION & EVALUATION
def generate_captions(model, processor, pixel_values, max_length=80, num_beams=4):
    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        length_penalty=LENGTH_PENALTY,
    )
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]


def filter_report_metrics(metric_results, split_name):
    ordered = MAIN_METRICS + EXTENDED_METRICS
    return {f"{split_name}_{k}": metric_results[k] for k in ordered}


def evaluate_model(model, loader, split_name="val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_refs, all_paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            total_loss += loss.item()
            preds = generate_captions(model, processor, pixel_values, GEN_MAX_LENGTH, NUM_BEAMS)
            refs = batch["captions"]

            all_preds.extend(preds)
            all_refs.extend(refs)
            all_paths.extend(batch["image_paths"])

    avg_loss = total_loss / max(len(loader), 1)
    metric_results = compute_overlap_and_semantic_metrics(all_preds, all_refs, all_paths)

    results = {
        f"{split_name}_loss": avg_loss,
        **filter_report_metrics(metric_results, split_name),
    }

    preview = []
    for img, pred, ref in list(zip(all_paths, all_preds, all_refs))[:10]:
        preview.append({"image": img, "prediction": pred, "reference": ref})

    return results, preview, all_preds, all_refs, all_paths


# TRAINING
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
best_val_score = -float("inf")
best_metric_name = PRIMARY_VAL_METRIC
history = []
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = running_loss / max(len(train_loader), 1)
    val_results, val_preview, _, _, _ = evaluate_model(model, val_loader, "val")

    epoch_log = {"epoch": epoch + 1, "train_loss": avg_train_loss, **val_results}

    current_val_score = float(val_results[PRIMARY_VAL_METRIC])
    improved = current_val_score > best_val_score + MIN_DELTA

    if improved:
        best_val_score = current_val_score
        epochs_without_improvement = 0

        save_dir = Path(OUTPUT_DIR) / "best_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

        print(f"Saved best model to {save_dir} using {PRIMARY_VAL_METRIC}={current_val_score:.6f}")
    else:
        epochs_without_improvement += 1

    epoch_log["monitor_metric"] = PRIMARY_VAL_METRIC
    epoch_log["monitor_value"] = current_val_score
    epoch_log["best_metric_name"] = best_metric_name
    epoch_log["best_val_score"] = best_val_score
    epoch_log["epochs_without_improvement"] = epochs_without_improvement

    history.append(epoch_log)
    print(json.dumps(epoch_log, indent=2))

    with open(Path(OUTPUT_DIR) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(Path(OUTPUT_DIR) / f"val_preview_epoch_{epoch + 1}.json", "w", encoding="utf-8") as f:
        json.dump(val_preview, f, indent=2, ensure_ascii=False)

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping triggered at epoch {epoch + 1}. "
            f"No improvement in {PRIMARY_VAL_METRIC} for {EARLY_STOPPING_PATIENCE} epoch(s)."
        )
        break

# TEST
best_model_dir = Path(OUTPUT_DIR) / "best_model"
processor = BlipProcessor.from_pretrained(best_model_dir)
model = BlipForConditionalGeneration.from_pretrained(best_model_dir).to(DEVICE)

test_results, test_preview, test_preds, test_refs, test_paths = evaluate_model(model, test_loader, "test")

test_results["selected_by_metric"] = PRIMARY_VAL_METRIC
test_results["selected_by_value"] = best_val_score

print("Test results:")
print(json.dumps(test_results, indent=2))

with open(Path(OUTPUT_DIR) / "test_results.json", "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=2, ensure_ascii=False)

with open(Path(OUTPUT_DIR) / "test_preview.json", "w", encoding="utf-8") as f:
    json.dump(test_preview, f, indent=2, ensure_ascii=False)

test_outputs = []
for img, pred, ref in zip(test_paths, test_preds, test_refs):
    test_outputs.append({"image": img, "prediction": pred, "reference": ref})

with open(Path(OUTPUT_DIR) / "test_predictions.json", "w", encoding="utf-8") as f:
    json.dump(test_outputs, f, indent=2, ensure_ascii=False)

with open(Path(OUTPUT_DIR) / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "prediction", "reference"])
    writer.writeheader()
    writer.writerows(test_outputs)


# INFERENCE HELPERS
def predict_caption(image_path, model, processor, device, max_length=80, num_beams=4):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            length_penalty=LENGTH_PENALTY,
        )

    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def predict_folder(image_dir, output_json, model, processor, device, max_length=80, num_beams=4):
    image_dir = Path(image_dir)
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    image_paths = sorted(
        [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
    )

    results = []
    for img_path in tqdm(image_paths, desc="Predicting captions"):
        cap = predict_caption(str(img_path), model, processor, device, max_length, num_beams)
        results.append({"image": str(img_path), "prediction": cap})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

# An example of using the predict_caption function on a single image. You can replace the path with any image you want to test.
example_image = "PMI_Class_Split/test/images/BLOAT/2012.035.09.07.2012 (12).jpg"
if os.path.exists(example_image):
    pred = predict_caption(example_image, model, processor, DEVICE, GEN_MAX_LENGTH, NUM_BEAMS)
    print("\nExample prediction:")
    print(example_image)
    print(pred)
