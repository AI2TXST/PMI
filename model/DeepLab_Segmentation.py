import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, ReLU, UpSampling2D,
    AveragePooling2D, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.list_physical_devices("GPU")
print("GPUs found:", gpus)

IMAGE_SIZE = 512
BATCH_SIZE = 8
TEST_BATCH_SIZE = 1
THRESHOLD = 0.5

TRAIN_IMAGE_FOLDER = Path("Segmentation/Train/Image")
TRAIN_MASK_FOLDER  = Path("Segmentation/Train/Masks")

VAL_IMAGE_FOLDER   = Path("Segmentation/Val/Image")
VAL_MASK_FOLDER    = Path("Segmentation/Val/Masks")

TEST_IMAGE_FOLDER  = Path("Segmentation/Test/Image")
TEST_MASK_FOLDER   = Path("Segmentation/Test/Masks")

MODEL_PATH = "./deeplab_final/Deeplab_PMI_Segmentation.keras"

PRED_MASK_DIR = Path("./deeplab_final/Predicted_Mask")
OVERLAY_DIR   = Path("./deeplab_final/Overlays")

PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_files(folder, exts):
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def build_pairs(image_folder, mask_folder):
    image_files = collect_files(image_folder, IMAGE_EXTS)
    mask_files = collect_files(mask_folder, MASK_EXTS)

    image_map = {p.stem: p for p in image_files}
    mask_map = {p.stem: p for p in mask_files}

    common_stems = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))

    if missing_images:
        print(f"[WARN] Masks without images in {mask_folder}: {len(missing_images)}")
        print(missing_images[:10])

    if missing_masks:
        print(f"[WARN] Images without masks in {image_folder}: {len(missing_masks)}")
        print(missing_masks[:10])

    image_paths = [str(image_map[s]) for s in common_stems]
    mask_paths  = [str(mask_map[s]) for s in common_stems]

    print(f"Matched pairs from {image_folder} and {mask_folder}: {len(common_stems)}")
    return image_paths, mask_paths


train_image_paths, train_mask_paths = build_pairs(TRAIN_IMAGE_FOLDER, TRAIN_MASK_FOLDER)
valid_image_paths, valid_mask_paths = build_pairs(VAL_IMAGE_FOLDER, VAL_MASK_FOLDER)
test_image_paths, test_mask_paths   = build_pairs(TEST_IMAGE_FOLDER, TEST_MASK_FOLDER)


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_image(image, channels=1, expand_animations=False)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.where(image >= 0.5, 1.0, 0.0)
        image = tf.image.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    else:
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            method=tf.image.ResizeMethod.BILINEAR
        )
        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    return image


def load_train_val_data(image_path, mask_path):
    image = read_image(image_path, mask=False)
    mask = read_image(mask_path, mask=True)
    return image, mask


def load_test_data_with_meta(image_path, mask_path):
    image_bytes = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image_decoded, tf.float32) / 255.0
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.BILINEAR
    )
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    mask_bytes = tf.io.read_file(mask_path)
    mask_original = tf.image.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask_original = tf.cast(mask_original, tf.float32) / 255.0
    mask_original = tf.where(mask_original >= 0.5, 1.0, 0.0)

    original_mask_h = tf.shape(mask_original)[0]
    original_mask_w = tf.shape(mask_original)[1]

    mask_resized = tf.image.resize(
        mask_original,
        [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    mask_resized = tf.where(mask_resized >= 0.5, 1.0, 0.0)
    mask_resized.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])

    original_mask_size = tf.stack([original_mask_h, original_mask_w])

    return image, mask_resized, mask_original, image_path, mask_path, original_mask_size


def augment_data(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)

    image = tf.clip_by_value(image, 0.0, 1.0)
    mask = tf.where(mask >= 0.5, 1.0, 0.0)

    return image, mask


def data_generator(image_list, mask_list, augment=False, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_train_val_data, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def data_generator_test(image_list, mask_list, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_test_data_with_meta, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = data_generator(
    train_image_paths, train_mask_paths, augment=True, batch_size=BATCH_SIZE
)
val_dataset = data_generator(
    valid_image_paths, valid_mask_paths, augment=False, batch_size=BATCH_SIZE
)
test_dataset = data_generator_test(
    test_image_paths, test_mask_paths, batch_size=TEST_BATCH_SIZE
)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)
print("Test Dataset:", test_dataset)


@keras.utils.register_keras_serializable(package="Custom", name="ConvBlock")
class ConvBlock(Layer):
    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.conv = Conv2D(
            filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
        })
        return config


def AtrousSpatialPyramidPooling(x):
    h = x.shape[1]
    w = x.shape[2]

    image_pool = AveragePooling2D(pool_size=(h, w))(x)
    image_pool = ConvBlock(kernel_size=1)(image_pool)
    image_pool = UpSampling2D(
        size=(h // image_pool.shape[1], w // image_pool.shape[2]),
        interpolation="bilinear"
    )(image_pool)

    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1)(x)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6)(x)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12)(x)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18)(x)

    x = layers.Concatenate(axis=-1)([image_pool, conv_1, conv_6, conv_12, conv_18])
    x = ConvBlock(kernel_size=1)(x)
    return x


def build_deeplabv3plus(image_size):
    input_layer = Input(shape=(image_size, image_size, 3))

    backbone = keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=input_layer
    )

    backbone_output = backbone.get_layer("block13_sepconv2_bn").output
    x = AtrousSpatialPyramidPooling(backbone_output)

    low_level = backbone.get_layer("block4_sepconv2_bn").output
    low_level = ConvBlock(filters=48, kernel_size=1)(low_level)

    x = UpSampling2D(
        size=(
            low_level.shape[1] // x.shape[1],
            low_level.shape[2] // x.shape[2],
        ),
        interpolation="bilinear"
    )(x)

    x = layers.Concatenate(axis=-1)([x, low_level])
    x = ConvBlock()(x)
    x = ConvBlock()(x)

    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear"
    )(x)

    output_layer = Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    return Model(inputs=input_layer, outputs=output_layer, name="DeepLabV3Plus_Xception")


@keras.saving.register_keras_serializable(package="Custom", name="dice_coef")
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


@keras.saving.register_keras_serializable(package="Custom", name="dice_loss")
def dice_loss(y_true, y_pred, smooth=1.0):
    return 1.0 - dice_coef(y_true, y_pred, smooth=smooth)


@keras.saving.register_keras_serializable(package="Custom", name="bce_dice_loss")
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    return bce + dloss


model = build_deeplabv3plus(IMAGE_SIZE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, "accuracy"],
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_dice_coef",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="max",
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    callbacks=[checkpoint],
    epochs=100,
)


m1 = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss,
        "ConvBlock": ConvBlock,
    }
)

val_score = m1.evaluate(val_dataset, verbose=0)
print("Validation results:", val_score)


def calculate_metrics(model, test_dataset, threshold=0.5, print_per_image=True):
    total_tp = 0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0

    per_image_results = []
    sample_idx = 0

    for images, masks_resized, masks_original, image_paths, mask_paths, original_mask_sizes in test_dataset:
        pred_probs = model.predict(images, verbose=0)
        masks_original_np = masks_original.numpy()
        batch_size = pred_probs.shape[0]

        for i in range(batch_size):
            orig_h = int(original_mask_sizes[i][0].numpy())
            orig_w = int(original_mask_sizes[i][1].numpy())

            pred_prob_i = pred_probs[i, ..., 0]
            gt_i = masks_original_np[i, :orig_h, :orig_w, 0]
            gt_i = (gt_i > 0.5).astype(np.uint8)

            pred_prob_resized = tf.image.resize(
                pred_prob_i[..., None],
                [orig_h, orig_w],
                method=tf.image.ResizeMethod.BILINEAR
            ).numpy()[..., 0]

            pred_i = (pred_prob_resized > threshold).astype(np.uint8)

            gt_flat = gt_i.flatten()
            pred_flat = pred_i.flatten()

            tp = np.sum((gt_flat == 1) & (pred_flat == 1)).astype(np.float64)
            tn = np.sum((gt_flat == 0) & (pred_flat == 0)).astype(np.float64)
            fp = np.sum((gt_flat == 0) & (pred_flat == 1)).astype(np.float64)
            fn = np.sum((gt_flat == 1) & (pred_flat == 0)).astype(np.float64)

            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            denom = np.sqrt(
                np.float64(tp + fp) *
                np.float64(tp + fn) *
                np.float64(tn + fp) *
                np.float64(tn + fn)
            ) + 1e-8

            mcc = ((tp * tn) - (fp * fn)) / denom

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            image_name = Path(image_paths[i].numpy().decode("utf-8")).name
            mask_name = Path(mask_paths[i].numpy().decode("utf-8")).name

            per_image_results.append({
                "image_index": sample_idx,
                "image_name": image_name,
                "mask_name": mask_name,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "accuracy": float(acc),
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "mcc": float(mcc),
            })

            sample_idx += 1

    per_image_df = pd.DataFrame(per_image_results)

    if print_per_image:
        print("\nPer-image metrics:")
        print(per_image_df.to_string(index=False))

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    specificity = total_tn / (total_tn + total_fp + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    denom = np.sqrt(
        np.float64(total_tp + total_fp) *
        np.float64(total_tp + total_fn) *
        np.float64(total_tn + total_fp) *
        np.float64(total_tn + total_fn)
    ) + 1e-8

    mcc = ((total_tp * total_tn) - (total_fp * total_fn)) / denom

    print("\nGlobal metrics from total confusion matrix:")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Dice       : {dice:.4f}")
    print(f"IoU        : {iou:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"MCC        : {mcc:.4f}")

    return per_image_df


per_image_df = calculate_metrics(m1, test_dataset, threshold=THRESHOLD)


GREEN = np.array([0, 1, 0], dtype=np.float32)
RED   = np.array([1, 0, 0], dtype=np.float32)
BLUE  = np.array([0, 0, 1], dtype=np.float32)


def save_result_mask_only(model, test_dataset, out_dir, threshold=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    for images, masks_resized, masks_original, image_paths, mask_paths, original_mask_sizes in test_dataset:
        pred_probs = model.predict(images, verbose=0)
        masks_original_np = masks_original.numpy()
        batch_size = pred_probs.shape[0]

        for i in range(batch_size):
            orig_h = int(original_mask_sizes[i][0].numpy())
            orig_w = int(original_mask_sizes[i][1].numpy())

            pred_prob_i = pred_probs[i, ..., 0]
            gt_i = masks_original_np[i, :orig_h, :orig_w, 0]
            gt_i = (gt_i > 0.5).astype(np.uint8)

            pred_prob_resized = tf.image.resize(
                pred_prob_i[..., None],
                [orig_h, orig_w],
                method=tf.image.ResizeMethod.BILINEAR
            ).numpy()[..., 0]

            pred_i = (pred_prob_resized > threshold).astype(np.uint8)

            tp = (pred_i == 1) & (gt_i == 1)
            fp = (pred_i == 1) & (gt_i == 0)
            fn = (pred_i == 0) & (gt_i == 1)

            result_mask = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
            result_mask[tp] = GREEN
            result_mask[fp] = RED
            result_mask[fn] = BLUE

            mask_name = Path(mask_paths[i].numpy().decode("utf-8")).stem
            save_path = out_dir / f"{mask_name}_overlay.png"

            plt.imsave(save_path, result_mask)
            print(f"Saved overlay: {save_path}")
            saved_count += 1

    print(f"\nTotal overlays saved: {saved_count}")


save_result_mask_only(m1, test_dataset, OVERLAY_DIR, threshold=THRESHOLD)


def save_predicted_masks(model, test_dataset, output_dir, threshold=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    for images, masks_resized, masks_original, image_paths, mask_paths, original_mask_sizes in test_dataset:
        pred_probs = model.predict(images, verbose=0)
        batch_size = pred_probs.shape[0]

        for i in range(batch_size):
            orig_h = int(original_mask_sizes[i][0].numpy())
            orig_w = int(original_mask_sizes[i][1].numpy())

            pred_prob_i = pred_probs[i, ..., 0]

            pred_prob_resized = tf.image.resize(
                pred_prob_i[..., None],
                [orig_h, orig_w],
                method=tf.image.ResizeMethod.BILINEAR
            ).numpy()[..., 0]

            pred_bin = (pred_prob_resized > threshold).astype(np.uint8) * 255

            input_name = Path(image_paths[i].numpy().decode("utf-8")).stem
            save_path = output_dir / f"{input_name}.png"

            Image.fromarray(pred_bin).save(save_path)
            print(f"Saved predicted mask: {save_path}")
            saved_count += 1

    print(f"\nTotal saved masks: {saved_count}")


save_predicted_masks(m1, test_dataset, PRED_MASK_DIR, threshold=THRESHOLD)
