#!/usr/bin/env python3
"""
train_fashion_mnist.py
======================
Trains a MobileNetV2-based transfer-learning model on **two** clothing
datasets and exports it as a TensorFlow Lite (TFLite) file ready to be
bundled inside the Flutter app.

Datasets
--------
1. Fashion-MNIST (built-in, downloaded automatically)
   - 70 000 grayscale 28×28 images.
   - 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal,
     Shirt, Sneaker, Bag, Ankle boot.
   - Strength: perfectly balanced (6 000 samples/class), canonical benchmark.
   - Limitation: no color information.

2. Kaggle Fashion Product Images – Small  (optional, pass --product-dataset-dir)
   Dataset page: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
   License: Apache 2.0 (commercial use permitted).
   - ~44 000 real product photos in full RGB color.
   - CSV columns used: id, articleType, baseColour.
   - articleType values are mapped to the same 10 Fashion-MNIST class indices
     via KAGGLE_ARTICLE_TYPE_TO_CLASS so both datasets share one output head.
   - Strength: teaches the backbone to recognise clothing in *color* photos,
     which is much closer to what the Flutter camera app captures at inference
     time than the greyscale Fashion-MNIST images.

   How to obtain the dataset
   -------------------------
   Option A – Kaggle CLI (requires a free Kaggle account):
       pip install kaggle
       kaggle datasets download -d paramaggarwal/fashion-product-images-small
       unzip fashion-product-images-small.zip -d fashion-product-images-small

   Option B – direct browser download from
       https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
   then unzip to a local directory.

   The directory must contain:
       <dir>/images/      ← one JPEG per product (named <id>.jpg)
       <dir>/styles.csv   ← CSV with id, articleType, baseColour, …

   Then pass:
       python train_fashion_mnist.py --product-dataset-dir <dir>

Why combine both datasets?
--------------------------
- Fashion-MNIST alone gives ~87 % accuracy but all images are grayscale; the
  model never sees color information during training.
- Kaggle real photos alone are insufficient: the dataset is imbalanced and
  some categories have too few samples.
- Together: Fashion-MNIST provides a balanced shape-signal foundation, while
  the Kaggle photos teach the backbone to extract features from real-world
  color photographs.  The result is a model that generalises better to the
  live camera feed in the app.

Memory note
-----------
Each preprocessed image occupies 128 × 128 × 3 × 4 bytes ≈ 192 KB.
• Fashion-MNIST training split (54 000 images):  ~10 GB
• Kaggle photos (default cap 20 000 images):      ~ 3.7 GB
• Combined:                                       ~13.7 GB
A machine with 16 GB RAM is recommended when using both datasets.
Use --max-product-samples to reduce the Kaggle sample count if memory is tight.

Architecture
------------
MobileNetV2 (ImageNet pre-trained, include_top=False)
  → GlobalAveragePooling2D → BN → Dense(512) → Dropout(0.3)
  → Dense(256) → Dropout(0.3) → Dense(10, softmax)

Training strategy
-----------------
Phase 1 (head only, backbone frozen):  up to PHASE1_MAX_EPOCHS epochs.
Phase 2 (backbone top-130 unfrozen):   at least 30 epochs with LR=1e-4.
Phase 3 (Kaggle color fine-tuning):    up to --kaggle-epochs, LR=5e-5.

Usage
-----
  # Fashion-MNIST only:
  python train_fashion_mnist.py [--epochs N] [--output PATH] [--no-augment]
                                [--phase2-lr F] [--fine-tune-layers N]

  # Fashion-MNIST + Kaggle color photos:
  python train_fashion_mnist.py --product-dataset-dir /path/to/fashion-product-images-small

Requirements
------------
  See requirements.txt
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ── Constants ────────────────────────────────────────────────────────────────

# Using 128×128 instead of the previous 96×96 gives MobileNetV2 more spatial
# detail to work with, improving fine-grained feature extraction for clothing.
# The trade-off is slightly longer training and a ~35 % larger input tensor,
# which is still well within mobile memory budgets.
MODEL_INPUT_SIZE: int = 128

# Number of input color channels (3 = RGB, matching MobileNetV2 expectations).
# Using RGB instead of grayscale also enables future color recognition.
MODEL_INPUT_CHANNELS: int = 3

# Number of output classes (one per Fashion-MNIST clothing category).
NUMBER_OF_CLOTHING_CLASSES: int = 10

# Default training hyper-parameters – override via command-line flags.
DEFAULT_TRAINING_EPOCHS: int = 50       # Maximum epochs (EarlyStopping kicks in)
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_VALIDATION_SPLIT: float = 0.1  # 10 % of training data for validation

# Maximum number of epochs for phase 1 (head-only training).
# The remainder of DEFAULT_TRAINING_EPOCHS is used for phase 2 fine-tuning.
PHASE1_MAX_EPOCHS: int = 20

# Fine-tuning: unfreeze this many layers from the *top* of the MobileNetV2
# backbone during phase 2.  130/154 layers lets more of the backbone adapt to
# clothing-specific textures while the lower 24 layers (generic edges/gradients)
# remain frozen and stable.
FINE_TUNE_LAYER_COUNT: int = 130

# Learning rate used during Phase 2 fine-tuning.
# 1e-4 is ~10× smaller than Phase-1 (3e-3) but high enough to actually adapt
# the backbone — the previous 5e-5 caused the loss to stall at ~1.9.
# Range recommended by literature for MobileNetV2 fine-tuning: 1e-4 – 2e-4.
DEFAULT_PHASE2_LR: float = 1e-4

# Label-smoothing coefficient applied to the cross-entropy loss.
# A value of 0.05 (reduced from 0.1) still prevents overconfidence while
# letting the model form sharper class boundaries – important when we need
# to distinguish visually similar items (shirt vs T-shirt).
LABEL_SMOOTHING: float = 0.05

# Maximum Kaggle samples **per class** to prevent bag-category domination.
# The Kaggle mapping has 12 bag subcategories vs 3 for shirts and 7 for
# trousers.  Without this cap, bag images dominate and the model predicts
# "Bag" for almost everything.  2000 samples/class gives a fair competition.
MAX_SAMPLES_PER_CLASS_KAGGLE: int = 2000

# Focal loss gamma parameter (Lin et al., 2017).
# gamma=2 is the standard value: easy, high-confidence predictions receive
# near-zero gradient while hard misclassified examples drive the update.
# This is the most principled solution to training on imbalanced datasets.
FOCAL_LOSS_GAMMA: float = 2.0

# ── Kaggle Fashion Product Images – article-type → class index mapping ────────
#
# The Kaggle "Fashion Product Images (Small)" dataset
# (https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
# contains a styles.csv with an `articleType` column.  The values below are
# mapped to the same 10 class indices used by Fashion-MNIST so both datasets
# share a single classification head.
#
# Fashion-MNIST class index reference:
#   0  T-shirt/top   1  Trouser   2  Pullover   3  Dress    4  Coat
#   5  Sandal        6  Shirt     7  Sneaker     8  Bag      9  Ankle boot
#
# Article types that do not map to any clothing class (e.g. Watches, Perfume,
# Jewellery) are intentionally omitted and will be filtered out during loading.
KAGGLE_ARTICLE_TYPE_TO_CLASS: dict[str, int] = {
    # ── Class 0: T-shirt/top ─────────────────────────────────────────────────
    "Tshirts":              0,
    "Tops":                 0,
    "Tank Tops":            0,
    "Tunics":               0,
    "Polo Tshirts":         0,
    "Innerwear Vests":      0,
    # ── Class 1: Trouser ─────────────────────────────────────────────────────
    "Jeans":                1,
    "Trousers":             1,
    "Track Pants":          1,
    "Shorts":               1,
    "Leggings":             1,
    "Capris":               1,
    "Tights":               1,
    # ── Class 2: Pullover ────────────────────────────────────────────────────
    "Sweatshirts":          2,
    "Sweaters":             2,
    "Hoodie":               2,
    "Nehru Jackets":        2,  # mandarin-collar knit tops
    # ── Class 3: Dress ───────────────────────────────────────────────────────
    "Dresses":              3,
    "Skirts":               3,
    "Kurtas":               3,
    "Sarees":               3,
    "Nightdress":           3,
    "Salwar":               3,
    "Dupatta":              3,
    # ── Class 4: Coat ────────────────────────────────────────────────────────
    "Jackets":              4,
    "Blazers":              4,
    "Coats":                4,
    "Windcheater":          4,
    "Rain Jacket":          4,
    "Waistcoat":            4,
    "Bomber Jackets":       4,
    "Dungarees":            4,
    # ── Class 5: Sandal ──────────────────────────────────────────────────────
    "Sandals":              5,
    "Heels":                5,
    "Flats":                5,
    "Flip Flops":           5,
    "Mules/Clogs":          5,
    # ── Class 6: Shirt ───────────────────────────────────────────────────────
    "Shirts":               6,
    "Casual Shirts":        6,
    "Formal Shirts":        6,
    # ── Class 7: Sneaker ─────────────────────────────────────────────────────
    "Sports Shoes":         7,
    "Casual Shoes":         7,
    "Sneakers":             7,
    "Running Shoes":        7,
    "Loafers":              7,
    "Formal Shoes":         7,
    # ── Class 8: Bag ─────────────────────────────────────────────────────────
    "Handbags":             8,
    "Backpacks":            8,
    "Clutches":             8,
    "Wallets":              8,
    "Laptop Bag":           8,
    "Trolley Bag":          8,
    "Messenger Bag":        8,
    "Rucksacks":            8,
    "Tote Bag":             8,
    "Sports Bag":           8,
    "Sling Bag":            8,
    "Duffle Bag":           8,
    # ── Class 9: Ankle boot ──────────────────────────────────────────────────
    "Boots":                9,
    "Ankle Boots":          9,
}

# Where to write the finished TFLite model file.
DEFAULT_TFLITE_OUTPUT_PATH: str = os.path.join(
    os.path.dirname(__file__),
    "..",
    "flutter_app",
    "assets",
    "models",
    "fashion_mnist_model.tflite",
)


# ── Custom losses & layers ────────────────────────────────────────────────────

class FocalLoss(keras.losses.Loss):
    """Focal loss for imbalanced datasets and hard-example mining.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    The (1 - p_t)^gamma factor reduces the contribution of easy, high-
    confidence predictions (p_t close to 1) and focuses learning on hard
    misclassified examples.  Combined with label smoothing this is much more
    effective than plain cross-entropy when training on imbalanced datasets
    where "Bag" would otherwise dominate.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = FOCAL_LOSS_GAMMA,
        label_smoothing: float = LABEL_SMOOTHING,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        # Apply label smoothing: prevents overconfident predictions.
        y_true_smooth = (
            y_true * (1.0 - self.label_smoothing)
            + self.label_smoothing / num_classes
        )
        # Clip to prevent log(0) NaN.
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Per-class cross-entropy weighted by focal factor.
        cross_entropy = -y_true_smooth * tf.math.log(y_pred_clipped)
        focal_weight = tf.pow(1.0 - y_pred_clipped, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))

    def get_config(self) -> dict:
        base = super().get_config()
        base.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return base


class RandomColorJitter(keras.layers.Layer):
    """Random hue and saturation shifts for MobileNetV2-normalised images.

    Images arrive in [-1, 1] (MobileNetV2 format).  The layer temporarily
    maps to [0, 1], applies tf.image random_hue / random_saturation, then
    maps back.  Only active during training=True so the augmentation is
    transparent at inference time and does not affect TFLite export.

    This teaches the backbone that clothing type is independent of color:
    bags appear in all colors, T-shirts appear in all colors, etc.  Without
    color jitter, the model may associate a particular hue with one class.
    """

    def __init__(
        self,
        hue_delta: float = 0.08,
        saturation_lower: float = 0.7,
        saturation_upper: float = 1.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hue_delta = hue_delta
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper

    def _apply_jitter(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply hue/saturation jitter (called inside tf.cond)."""
        # [-1, 1] → [0, 1] for tf.image operations.
        x = (inputs + 1.0) / 2.0
        x = tf.image.random_hue(x, self.hue_delta)
        x = tf.image.random_saturation(x, self.saturation_lower, self.saturation_upper)
        x = tf.clip_by_value(x, 0.0, 1.0)
        # [0, 1] → [-1, 1] back to MobileNetV2 range.
        return x * 2.0 - 1.0

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        # Use tf.cond so this branch is graph-compiled under tf.function.
        # A plain Python `if not training` would be frozen at trace time and
        # always skip the augmentation during fine-tuning.
        training_flag = tf.cast(
            training if training is not None else False, tf.bool
        )
        return tf.cond(
            training_flag,
            lambda: self._apply_jitter(inputs),
            lambda: inputs,
        )

    def get_config(self) -> dict:
        base = super().get_config()
        base.update({
            "hue_delta": self.hue_delta,
            "saturation_lower": self.saturation_lower,
            "saturation_upper": self.saturation_upper,
        })
        return base


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_preprocess_fashion_mnist_dataset() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Load the full Fashion-MNIST dataset and prepare it for MobileNetV2.

    Pipeline per image
    ------------------
    1. Resize 28×28 → MODEL_INPUT_SIZE×MODEL_INPUT_SIZE using bilinear
       interpolation (tf.image.resize).
    2. Repeat the single grayscale channel 3× to form a pseudo-RGB tensor.
    3. Apply MobileNetV2's preprocess_input (maps [0, 255] → [-1.0, 1.0]).
    4. Convert integer labels to one-hot vectors for label-smoothing support.

    The Flutter app's ImagePreprocessingService applies the *same* three steps
    so training and inference are perfectly aligned.

    Returns
    -------
    training_images       : np.ndarray  shape (60 000, 128, 128, 3), float32
    training_labels_onehot: np.ndarray  shape (60 000, 10), float32
    test_images           : np.ndarray  shape (10 000, 128, 128, 3), float32
    test_labels_onehot    : np.ndarray  shape (10 000, 10), float32
    """
    # Keras downloads and caches the dataset automatically.
    (raw_train_images, training_labels), (raw_test_images, test_labels) = (
        keras.datasets.fashion_mnist.load_data()
    )

    def _preprocess(images: np.ndarray) -> np.ndarray:
        """Resize → pseudo-RGB → MobileNetV2 normalisation."""
        # Add channel dim: (N, 28, 28) → (N, 28, 28, 1)
        images = np.expand_dims(images.astype("float32"), axis=-1)

        # Resize to MODEL_INPUT_SIZE using TensorFlow ops (fast, batched).
        images = tf.image.resize(
            images,
            [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            method=tf.image.ResizeMethod.BILINEAR,
        ).numpy()

        # Repeat grayscale channel: (N, H, W, 1) → (N, H, W, 3)
        images = np.repeat(images, MODEL_INPUT_CHANNELS, axis=-1)

        # Apply MobileNetV2 standard preprocessing.
        # preprocess_input expects float values in [0, 255] and maps them
        # to [-1.0, 1.0] via:  x / 127.5 - 1.0
        # The values are in [0.0, 255.0] at this point because astype("float32")
        # preserves the original uint8 range – tf.image.resize only resamples
        # spatially and does not rescale pixel values.
        images = preprocess_input(images)
        return images

    print(f"      Preprocessing images to {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}×{MODEL_INPUT_CHANNELS}…")
    training_images = _preprocess(raw_train_images)
    test_images = _preprocess(raw_test_images)

    # Convert integer class labels to one-hot vectors so that label smoothing
    # can be applied via CategoricalCrossentropy during training.
    # Shape: (N,) → (N, NUMBER_OF_CLOTHING_CLASSES)
    training_labels_onehot = keras.utils.to_categorical(
        training_labels, num_classes=NUMBER_OF_CLOTHING_CLASSES
    )
    test_labels_onehot = keras.utils.to_categorical(
        test_labels, num_classes=NUMBER_OF_CLOTHING_CLASSES
    )

    return (
        training_images, training_labels_onehot,
        test_images, test_labels_onehot,
    )


# ── Kaggle Fashion Product Images loader ──────────────────────────────────────

def load_and_preprocess_fashion_product_dataset(
    dataset_dir: str,
    max_samples: int | None = None,
    max_per_class: int | None = MAX_SAMPLES_PER_CLASS_KAGGLE,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and preprocess the Kaggle Fashion Product Images (Small) dataset.

    The function reads ``styles.csv``, filters rows whose ``articleType`` is
    in KAGGLE_ARTICLE_TYPE_TO_CLASS, **balances samples per class** to prevent
    bag-category domination, loads the corresponding JPEG images, and applies
    the same preprocessing pipeline as Fashion-MNIST (resize → RGB →
    MobileNetV2 normalisation) so both datasets are compatible.

    Why per-class balancing is critical
    ------------------------------------
    The Kaggle mapping contains 12 bag subcategories (Handbags, Backpacks,
    Wallets, …) vs only 3 for shirts.  Without capping, bag images outnumber
    shirt images by ~4×, and the model learns to predict "Bag" for anything
    it is uncertain about.  ``max_per_class`` enforces a fair per-class limit
    before the total ``max_samples`` cap is applied.

    Dataset page
    ------------
    https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
    License: Apache 2.0 (commercial use permitted).

    Expected directory layout
    -------------------------
    <dataset_dir>/
      images/         ← JPEG files named <id>.jpg
      styles.csv      ← CSV with columns: id, articleType, baseColour, …

    Parameters
    ----------
    dataset_dir : str
        Path to the extracted Kaggle dataset directory.
    max_samples : int | None
        Hard cap on total samples after per-class balancing (memory guard).
    max_per_class : int | None
        Maximum images **per class** (default: MAX_SAMPLES_PER_CLASS_KAGGLE).
        Set to None to disable per-class balancing (not recommended).

    Returns
    -------
    (images, labels_onehot) : tuple[np.ndarray, np.ndarray]
        images        : float32 array of shape (N, 128, 128, 3), MobileNetV2
                        normalised, full RGB color.
        labels_onehot : float32 array of shape (N, 10), one-hot encoded using
                        the same class ordering as Fashion-MNIST.
    None
        Returned when ``dataset_dir`` does not contain the expected files.
    """
    csv_path = os.path.join(dataset_dir, "styles.csv")
    images_dir = os.path.join(dataset_dir, "images")

    if not os.path.isfile(csv_path):
        print(f"      [WARNING] styles.csv not found in {dataset_dir!r}. "
              "Skipping Kaggle dataset.")
        return None
    if not os.path.isdir(images_dir):
        print(f"      [WARNING] images/ directory not found in {dataset_dir!r}. "
              "Skipping Kaggle dataset.")
        return None

    # ── Read and filter the CSV ───────────────────────────────────────────
    # Some rows have malformed trailing data; on_bad_lines='skip' (pandas ≥ 2.0)
    # silently drops those rows instead of raising an error.
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # Keep only rows whose articleType maps to one of our 10 classes.
    df = df[df["articleType"].isin(KAGGLE_ARTICLE_TYPE_TO_CLASS)].copy()
    df["class_index"] = df["articleType"].map(KAGGLE_ARTICLE_TYPE_TO_CLASS)

    # ── Per-class balancing (CRITICAL: prevents bag domination) ──────────
    # Cap each class independently so no single class (especially Bag with
    # 12 subcategories) can overwhelm the training signal.
    if max_per_class is not None:
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
        balanced_parts = []
        print("      Per-class distribution before balancing:")
        for class_idx in range(NUMBER_OF_CLOTHING_CLASSES):
            class_df = df[df["class_index"] == class_idx]
            raw_count = len(class_df)
            if raw_count > max_per_class:
                class_df = class_df.sample(n=max_per_class, random_state=42)
            capped = len(class_df)
            name = class_names[class_idx]
            print(f"        {class_idx}. {name:12s}: {raw_count:5,} raw → {capped:5,} kept")
            balanced_parts.append(class_df)
        df = (
            pd.concat(balanced_parts)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    # Hard total-sample cap (memory guard applied after per-class balancing).
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    print(f"      Kaggle rows after balancing: {len(df):,}")
    if len(df) == 0:
        print("      [WARNING] No usable rows found. Skipping Kaggle dataset.")
        return None

    # ── Load images ───────────────────────────────────────────────────────
    loaded_images: list[np.ndarray] = []
    loaded_labels: list[int] = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        image_path = os.path.join(images_dir, f"{int(row['id'])}.jpg")

        if not os.path.isfile(image_path):
            skipped += 1
            continue

        try:
            raw_bytes = tf.io.read_file(image_path)
            # decode_jpeg handles JPEG; channels=3 forces RGB (no alpha).
            image_tensor = tf.image.decode_jpeg(raw_bytes, channels=3)
            # Resize to the same spatial resolution as Fashion-MNIST images.
            image_tensor = tf.image.resize(
                image_tensor,
                [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                method=tf.image.ResizeMethod.BILINEAR,
            )
            # Apply MobileNetV2 normalisation ([0, 255] → [-1, 1]).
            image_np = preprocess_input(image_tensor.numpy().astype("float32"))
            loaded_images.append(image_np)
            loaded_labels.append(int(row["class_index"]))
        except Exception as exc:  # corrupted JPEG, permission error, etc.
            skipped += 1
            if skipped <= 5:  # avoid flooding the console
                print(f"      [WARNING] Could not load {image_path}: {exc}")
            continue

        # Progress update every 2 000 images.
        if (i + 1) % 2000 == 0:
            print(f"      Loaded {i + 1:,} / {len(df):,} images "
                  f"({skipped} skipped)…")

    print(f"      Finished loading: {len(loaded_images):,} images, "
          f"{skipped} skipped.")

    if len(loaded_images) == 0:
        print("      [WARNING] All images failed to load. "
              "Skipping Kaggle dataset.")
        return None

    images_array = np.stack(loaded_images, axis=0)  # (N, 128, 128, 3)
    labels_onehot = keras.utils.to_categorical(
        np.array(loaded_labels, dtype=np.int32),
        num_classes=NUMBER_OF_CLOTHING_CLASSES,
    )  # (N, 10)

    return images_array, labels_onehot


# ── Class weight computation ──────────────────────────────────────────────────

_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def compute_class_weights(labels_onehot: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights from one-hot training labels.

    Weight formula:  weight_i = N / (C * count_i)
    where N = total samples, C = number of classes, count_i = class i count.

    Under-represented classes (e.g. Shirt with few Kaggle samples) receive
    a higher weight so that every misclassification of a shirt hurts as much
    as misclassifying a bag, even if bags are 3× more common.

    Parameters
    ----------
    labels_onehot : np.ndarray  Shape (N, 10).

    Returns
    -------
    dict[int, float]  Mapping from class index to sample weight.
    """
    labels_int = np.argmax(labels_onehot, axis=1)
    class_counts = np.bincount(labels_int, minlength=NUMBER_OF_CLOTHING_CLASSES)
    total = len(labels_int)
    raw_weights = total / (NUMBER_OF_CLOTHING_CLASSES * class_counts.astype(float))

    print("      Combined dataset class distribution:")
    for i, (count, w) in enumerate(zip(class_counts, raw_weights)):
        bar = "#" * min(count // 2000, 25)
        print(f"        {i}. {_CLASS_NAMES[i]:12s}: {count:6,} samples  weight={w:.3f}  {bar}")

    return {i: float(w) for i, w in enumerate(raw_weights)}


# ── Data augmentation ─────────────────────────────────────────────────────────

def build_augmentation_pipeline() -> keras.Sequential:
    """On-the-fly data augmentation applied only during training.

    Augmentations simulate real camera variation:
    - Horizontal flip: clothing is left-right symmetric.
    - Rotation ±10°, zoom ±10%, translation ±8%: different shooting angles.
    - Brightness/contrast ±15%: varying lighting conditions.
    - RandomColorJitter (hue ±0.05, saturation 0.8–1.2): the same clothing
      type appears in many colors; color jitter prevents the model from
      linking a specific hue to a specific class (crucial for real-world use).

    Vertical flips are excluded because clothing is always right-side up.
    All transforms are applied with tf.keras seed-safe ops, so training is
    fully reproducible when a global seed is set.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.028),      # ±10° (was ±36°)
            keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            keras.layers.RandomTranslation(
                height_factor=0.08, width_factor=0.08
            ),
            keras.layers.RandomBrightness(factor=0.15),
            keras.layers.RandomContrast(factor=0.15),
            RandomColorJitter(
                hue_delta=0.05,
                saturation_lower=0.8,
                saturation_upper=1.2,
                name="color_jitter",
            ),
        ],
        name="data_augmentation",
    )


# ── Model architecture ────────────────────────────────────────────────────────

def build_clothing_classifier_model(use_augmentation: bool = True) -> tuple[keras.Model, keras.Model]:
    """Build a MobileNetV2 transfer-learning clothing classifier.

    Architecture overview
    ---------------------
    Input  128×128×3 (pseudo-RGB, MobileNetV2 normalised)
      │
      ├─ [Optional] Data augmentation pipeline
      ├─ MobileNetV2 backbone (pre-trained on ImageNet, include_top=False)
      │    └─ outputs 4×4×1280 feature map at 128×128 input
      ├─ GlobalAveragePooling2D  →  1280-D feature vector
      ├─ BatchNormalization
      ├─ Dense(512, relu)       ← wider head for more representational capacity
      ├─ Dropout(0.3)
      ├─ Dense(256, relu)
      ├─ Dropout(0.3)
      └─ Dense(10, softmax)   ← one neuron per Fashion-MNIST class

    Training strategy
    -----------------
    Phase 1 (head training):
        Freeze the entire MobileNetV2 backbone.  Train only the custom head
        for a few epochs so the new dense layers converge before the backbone
        weights are touched.

    Phase 2 (fine-tuning):
        Unfreeze the top FINE_TUNE_LAYER_COUNT layers of MobileNetV2 and
        continue training with a much smaller learning rate.  Lower layers
        retain ImageNet features; upper layers adapt to clothing-specific
        textures and shapes.

    Label smoothing (0.05) is built into FocalLoss so the model is penalised
    for being overconfident, improving generalisation.

    Extensibility
    -------------
    To add color or pattern recognition, attach additional output heads to
    the GlobalAveragePooling2D layer or load the same MobileNetV2 backbone
    in a separate model.  No changes to this function are needed.

    Returns
    -------
    keras.Model
        Compiled model (phase-1 configuration, backbone frozen).
    """
    input_layer = keras.Input(
        shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS),
        name="rgb_image_input",
    )

    x = input_layer

    # ── Optional data augmentation (active only during training) ─────────
    if use_augmentation:
        x = build_augmentation_pipeline()(x)

    # ── MobileNetV2 backbone ──────────────────────────────────────────────
    # include_top=False removes the ImageNet classification head.
    # weights='imagenet' loads the pre-trained weights.
    # The backbone is created outside the functional API and then called as
    # a layer so that the trainable flag can be toggled independently.
    mobilenet_backbone = MobileNetV2(
        input_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    # Freeze all backbone layers for phase 1 (head-only training).
    mobilenet_backbone.trainable = False

    x = mobilenet_backbone(x, training=False)

    # ── Custom classification head ─────────────────────────────────────────
    # A wider first Dense layer (512 vs the previous 256) gives the model
    # more capacity to combine the 1280-D backbone features before classifying.
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dense(units=512, activation="relu", name="dense_features_1")(x)
    x = keras.layers.Dropout(rate=0.3, name="dropout_1")(x)
    x = keras.layers.Dense(units=256, activation="relu", name="dense_features_2")(x)
    x = keras.layers.Dropout(rate=0.3, name="dropout_2")(x)

    output_probabilities = keras.layers.Dense(
        units=NUMBER_OF_CLOTHING_CLASSES,
        activation="softmax",
        name="clothing_class_probabilities",
    )(x)

    model = keras.Model(
        inputs=input_layer,
        outputs=output_probabilities,
        name="mobilenetv2_clothing_classifier",
    )

    # Compile for phase 1.
    # FocalLoss focuses the gradient on hard, misclassified examples and
    # handles class imbalance far better than plain CategoricalCrossentropy.
    # Label smoothing (0.05) is built into FocalLoss to prevent overconfidence.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-3),
        loss=FocalLoss(),
        metrics=["accuracy"],
    )

    return model, mobilenet_backbone


# ── Training ──────────────────────────────────────────────────────────────────

def _make_callbacks(
    patience_reduce: int = 3,
    patience_stop: int = 6,
    monitor: str = "val_loss",
) -> list:
    """Return standard ReduceLROnPlateau + EarlyStopping callbacks."""
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.3,        # less aggressive than 0.5: 0.3× instead of 0.5×
            patience=patience_reduce,
            min_lr=1e-6,       # floor raised slightly to avoid near-zero updates
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def train_model_phase1(
    model: keras.Model,
    training_images: np.ndarray,
    training_labels_onehot: np.ndarray,
    number_of_epochs: int,
    class_weight: dict[int, float] | None = None,
) -> None:
    """Phase 1: train only the custom head (backbone is frozen).

    Parameters
    ----------
    model                   : keras.Model           Compiled model.
    training_images         : np.ndarray            Preprocessed images (N, 128, 128, 3).
    training_labels_onehot  : np.ndarray            One-hot labels (N, 10).
    number_of_epochs        : int                   Upper bound on training epochs.
    class_weight            : dict[int, float] | None  Per-class loss multipliers.
    """
    # Limit phase 1 to at most PHASE1_MAX_EPOCHS (EarlyStopping will cut it shorter).
    phase1_epochs = min(number_of_epochs, PHASE1_MAX_EPOCHS)

    print(f"\n  Phase 1 – training head only (up to {phase1_epochs} epochs)…")
    if class_weight:
        print("  Class weights: active (countering imbalance)")
    model.fit(
        training_images,
        training_labels_onehot,
        epochs=phase1_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        validation_split=DEFAULT_VALIDATION_SPLIT,
        class_weight=class_weight,
        callbacks=_make_callbacks(patience_reduce=3, patience_stop=5),
        verbose=1,
    )


def train_model_phase2(
    model: keras.Model,
    mobilenet_backbone: keras.Model,
    training_images: np.ndarray,
    training_labels_onehot: np.ndarray,
    number_of_epochs: int,
    phase2_lr: float = DEFAULT_PHASE2_LR,
    fine_tune_layers: int = FINE_TUNE_LAYER_COUNT,
    class_weight: dict[int, float] | None = None,
) -> None:
    """Phase 2: unfreeze the top layers of MobileNetV2 and fine-tune.

    The top fine_tune_layers layers are unfrozen; lower layers remain
    frozen so their low-level ImageNet features are preserved.  A very small
    learning rate (default 5e-5, 200× smaller than Phase-1) prevents large
    gradient updates from destroying the pre-trained weights.

    Parameters
    ----------
    model                   : keras.Model           The full model (head + backbone).
    mobilenet_backbone      : keras.Model           The MobileNetV2 sub-model to unfreeze.
    training_images         : np.ndarray            Same preprocessed images as phase 1.
    training_labels_onehot  : np.ndarray            Same one-hot labels as phase 1.
    number_of_epochs        : int                   Upper bound on total epochs.
    phase2_lr               : float                 Fine-tuning learning rate.
    fine_tune_layers        : int                   Backbone layers to unfreeze.
    class_weight            : dict[int, float] | None  Per-class loss multipliers.
    """
    # Unfreeze everything, then re-freeze the lower layers.
    mobilenet_backbone.trainable = True
    total_layers = len(mobilenet_backbone.layers)
    freeze_until = total_layers - fine_tune_layers
    for layer in mobilenet_backbone.layers[:freeze_until]:
        layer.trainable = False

    frozen_count = sum(1 for l in mobilenet_backbone.layers if not l.trainable)
    print(
        f"\n  Phase 2 – fine-tuning top {fine_tune_layers} of "
        f"{total_layers} backbone layers "
        f"({frozen_count} layers still frozen)…"
    )

    # Recompile with a much smaller LR so the backbone weights change slowly.
    # FocalLoss is kept identical to phase 1 for consistent training signal.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase2_lr),
        loss=FocalLoss(),
        metrics=["accuracy"],
    )

    # Guarantee at least 30 fine-tuning epochs (EarlyStopping provides the
    # safety net if the model stops improving).
    remaining_epochs = max(number_of_epochs - PHASE1_MAX_EPOCHS, 30)
    try:
        model.fit(
            training_images,
            training_labels_onehot,
            epochs=remaining_epochs,
            batch_size=DEFAULT_BATCH_SIZE,
            validation_split=DEFAULT_VALIDATION_SPLIT,
            class_weight=class_weight,
            callbacks=_make_callbacks(
                patience_reduce=5,
                patience_stop=10,
                monitor="val_loss",
            ),
            verbose=1,
        )
    except MemoryError:
        print(
            "\n  [WARNING] Not enough RAM for Phase-2 validation split. "
            "Retrying Phase 2 without validation data."
        )
        model.fit(
            training_images,
            training_labels_onehot,
            epochs=remaining_epochs,
            batch_size=DEFAULT_BATCH_SIZE,
            validation_split=0.0,
            class_weight=class_weight,
            callbacks=_make_callbacks(
                patience_reduce=5,
                patience_stop=10,
                monitor="loss",
            ),
            verbose=1,
        )


def train_model_phase3(
    model: keras.Model,
    mobilenet_backbone: keras.Model,
    kaggle_images: np.ndarray,
    kaggle_labels_onehot: np.ndarray,
    phase3_lr: float = 5e-5,
    kaggle_epochs: int = 20,
    class_weight: dict[int, float] | None = None,
) -> None:
    """Phase 3 (curriculum learning): fine-tune on real Kaggle color photos.

    After the model has learned robust shape features from Fashion-MNIST in
    phases 1 and 2, this phase exposes it to real camera photographs with
    full RGB color.  A small learning rate preserves the shape knowledge while
    adapting the backbone to color textures, real-world lighting, and
    perspective variation.

    Why curriculum learning works here
    ------------------------------------
    Fashion-MNIST is perfectly balanced, clean, and grayscale – ideal for
    learning the *shape* of each clothing category.  Kaggle photos are noisy,
    imbalanced, and in color – ideal for teaching the model to handle real
    photos.  Training on shapes first and adding color second is the same
    curriculum a human learner would follow.

    Parameters
    ----------
    model               : keras.Model           Fully trained model (after phase 2).
    mobilenet_backbone  : keras.Model           The MobileNetV2 sub-model (already unfrozen).
    kaggle_images       : np.ndarray            Kaggle photos (N, 128, 128, 3), float32.
    kaggle_labels_onehot: np.ndarray            One-hot labels (N, 10), float32.
    phase3_lr           : float                 LR for Kaggle fine-tuning (default 5e-5).
    kaggle_epochs       : int                   Max fine-tuning epochs on Kaggle data.
    class_weight        : dict[int, float] | None  Per-class loss multipliers.
    """
    print(f"\n  Phase 3 – Kaggle color-photo fine-tuning "
          f"(up to {kaggle_epochs} epochs, lr={phase3_lr})…")
    print(f"      Kaggle training samples : {len(kaggle_images):,}")
    if class_weight:
        print("      Class weights: active")

    # The backbone is already partially unfrozen from phase 2; keep that
    # configuration and just lower the learning rate further.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase3_lr),
        loss=FocalLoss(),
        metrics=["accuracy"],
    )

    try:
        model.fit(
            kaggle_images,
            kaggle_labels_onehot,
            epochs=kaggle_epochs,
            batch_size=DEFAULT_BATCH_SIZE,
            validation_split=DEFAULT_VALIDATION_SPLIT,
            class_weight=class_weight,
            callbacks=_make_callbacks(
                patience_reduce=4,
                patience_stop=8,
                monitor="val_loss",
            ),
            verbose=1,
        )
    except MemoryError:
        print("\n  [WARNING] Not enough RAM for Phase-3 validation split. "
              "Retrying without validation data.")
        model.fit(
            kaggle_images,
            kaggle_labels_onehot,
            epochs=kaggle_epochs,
            batch_size=DEFAULT_BATCH_SIZE,
            validation_split=0.0,
            class_weight=class_weight,
            callbacks=_make_callbacks(
                patience_reduce=4,
                patience_stop=8,
                monitor="loss",
            ),
            verbose=1,
        )


# ── TFLite export ─────────────────────────────────────────────────────────────

def export_model_to_tflite(
    trained_model: keras.Model, tflite_output_path: str
) -> None:
    """Convert [trained_model] to TFLite format and save to disk.

    Dynamic range quantisation is applied: it compresses weights to int8 but
    keeps model inputs and outputs as float32.  The Flutter app therefore feeds
    a float32 tensor and receives float32 probabilities with no type mismatch.

    A quick sanity-check inference is run after conversion to verify that the
    exported model loads and executes correctly.

    Parameters
    ----------
    trained_model      : keras.Model   Fully trained model.
    tflite_output_path : str           Path where the .tflite file is written.
    """
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)

    # Dynamic range quantisation: weights int8, activations float32.
    # The Flutter app sends float32 input → no quantisation mismatch.
    tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model_bytes = tflite_converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(tflite_output_path)), exist_ok=True)

    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model_bytes)

    size_kb = os.path.getsize(tflite_output_path) / 1024
    print(f"TFLite model saved → {tflite_output_path} ({size_kb:.1f} KB)")

    # --- Sanity check: one forward pass with a blank image ----------------
    interpreter = tf.lite.Interpreter(model_path=tflite_output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy = np.zeros(input_details[0]["shape"], dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    print(
        f"Sanity check passed – "
        f"input shape: {input_details[0]['shape']}, "
        f"output shape: {output_details[0]['shape']}, "
        f"probability sum: {output.sum():.4f}"
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model_accuracy(
    trained_model: keras.Model,
    test_images: np.ndarray,
    test_labels_onehot: np.ndarray,
) -> None:
    """Print the test-set loss and accuracy of [trained_model]."""
    test_loss, test_accuracy = trained_model.evaluate(
        test_images, test_labels_onehot, verbose=0
    )
    print(f"Test accuracy : {test_accuracy * 100:.2f} %")
    print(f"Test loss     : {test_loss:.4f}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command-line flags and return the argument namespace."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune MobileNetV2 on Fashion-MNIST and export as TFLite."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS,
        help=f"Maximum training epochs across both phases (default: {DEFAULT_TRAINING_EPOCHS}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TFLITE_OUTPUT_PATH,
        help="Output path for the .tflite model file.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="Disable data augmentation (faster training, lower accuracy).",
    )
    parser.add_argument(
        "--phase2-lr",
        type=float,
        default=DEFAULT_PHASE2_LR,
        help=(
            f"Learning rate for Phase-2 fine-tuning (default: {DEFAULT_PHASE2_LR})."
            " Lower values produce more stable fine-tuning."
        ),
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=FINE_TUNE_LAYER_COUNT,
        help=(
            f"Number of top backbone layers to unfreeze in Phase 2 "
            f"(default: {FINE_TUNE_LAYER_COUNT}). "
            "Higher values allow deeper adaptation but require more memory."
        ),
    )
    parser.add_argument(
        "--product-dataset-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Path to the extracted Kaggle Fashion Product Images (Small) "
            "dataset directory.  The directory must contain images/ and "
            "styles.csv.  When provided, the Kaggle photos are combined with "
            "Fashion-MNIST for training, giving the model exposure to real "
            "color photographs.  Dataset: "
            "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small"
        ),
    )
    parser.add_argument(
        "--max-product-samples",
        type=int,
        default=20_000,
        metavar="N",
        help=(
            "Maximum number of Kaggle Fashion Product images to load "
            "(default: 20 000).  Reduce this value if you run out of RAM. "
            "Each image takes ~192 KB as a float32 tensor; 20 000 images "
            "≈ 3.7 GB.  Ignored when --product-dataset-dir is not set."
        ),
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=MAX_SAMPLES_PER_CLASS_KAGGLE,
        metavar="N",
        help=(
            "Maximum Kaggle images per class before combining with Fashion-MNIST "
            f"(default: {MAX_SAMPLES_PER_CLASS_KAGGLE}).  "
            "Prevents bag-category domination: the Kaggle dataset has 12 bag "
            "subcategories vs 3 for shirts, so without this cap the model learns "
            "to predict \"Bag\" for uncertain inputs."
        ),
    )
    parser.add_argument(
        "--phase3-lr",
        type=float,
        default=5e-5,
        help=(
            "Learning rate for Phase-3 Kaggle color fine-tuning (default: 5e-5). "
            "Should be smaller than --phase2-lr to preserve shape knowledge "
            "learned in Stage A."
        ),
    )
    parser.add_argument(
        "--kaggle-epochs",
        type=int,
        default=20,
        help=(
            "Maximum epochs for Phase-3 Kaggle fine-tuning (default: 20). "
            "EarlyStopping may cut this shorter if validation loss stops improving."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Curriculum learning pipeline:

    Stage A – Shape learning (Fashion-MNIST, grayscale, perfectly balanced)
      Phase 1: Train classification head only (backbone frozen).
      Phase 2: Fine-tune top backbone layers on Fashion-MNIST.

    Stage B – Real-photo adaptation (Kaggle color photos, optional)
      Phase 3: Fine-tune on Kaggle real photos to learn color + real-world
               appearance.  The shape knowledge from Stage A is preserved
               because the learning rate is kept very small.

    This curriculum mirrors how a human learner works: first understand
    *what* makes a trouser look like a trouser (shape), then learn that
    trousers come in many colors and lighting conditions (real photos).
    """
    args = parse_command_line_arguments()
    use_augmentation: bool = not args.no_augment
    fine_tune_layers: int = args.fine_tune_layers

    print("=" * 65)
    print("  Clothing Classifier – MobileNetV2 Curriculum Pipeline")
    print("=" * 65)
    print(f"  Input size       : {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}×{MODEL_INPUT_CHANNELS}")
    print(f"  Augmentation     : {'ON' if use_augmentation else 'OFF'}")
    print(f"  Max epochs       : {args.epochs} (phase 1 ≤{PHASE1_MAX_EPOCHS}, phase 2 ≥30)")
    print(f"  Fine-tune layers : {fine_tune_layers}")
    print(f"  Phase-2 LR       : {args.phase2_lr}")
    print(f"  Label smoothing  : {LABEL_SMOOTHING}")
    if args.product_dataset_dir:
        print(f"  Kaggle dataset   : {args.product_dataset_dir}")
        print(f"  Max product imgs : {args.max_product_samples:,}")
        print(f"  Max per class    : {args.max_per_class:,}")
        print(f"  Training stages  : Phase 1+2 (FMNIST shapes) → Phase 3 (Kaggle color)")
    else:
        print("  Kaggle dataset   : not used (pass --product-dataset-dir to enable)")
        print("  Training stages  : Phase 1+2 (FMNIST shapes only)")

    # ── Step 1: Load Fashion-MNIST ─────────────────────────────────────────
    print("\n[1/5] Loading & preprocessing Fashion-MNIST (all 70 000 samples)…")
    fm_train_images, fm_train_labels, test_images, test_labels_onehot = (
        load_and_preprocess_fashion_mnist_dataset()
    )
    print(f"      Fashion-MNIST training : {len(fm_train_images):,} samples")
    print(f"      Fashion-MNIST test     : {len(test_images):,} samples")

    # ── Step 2: Optionally load Kaggle color photos (kept separate) ────────
    # IMPORTANT: Kaggle data is NO LONGER merged with Fashion-MNIST.
    # It is used exclusively in Phase 3 so the model first learns shapes
    # from clean balanced data, then adapts to real photos.
    kaggle_images: np.ndarray | None = None
    kaggle_labels: np.ndarray | None = None

    if args.product_dataset_dir is not None:
        print(
            f"\n[2/5] Loading Kaggle Fashion Product Images "
            f"(max {args.max_product_samples:,} total, max {args.max_per_class:,} per class)…"
        )
        kaggle_result = load_and_preprocess_fashion_product_dataset(
            dataset_dir=args.product_dataset_dir,
            max_samples=args.max_product_samples,
            max_per_class=args.max_per_class,
        )
        if kaggle_result is not None:
            kaggle_images, kaggle_labels = kaggle_result
            print(f"      Kaggle samples loaded  : {len(kaggle_images):,}")
            print("      (Will be used in Phase 3 for color/real-world fine-tuning)")
        else:
            print("      Kaggle dataset skipped – will train on Fashion-MNIST only.")
    else:
        print("\n[2/5] Skipping Kaggle dataset (--product-dataset-dir not set).")

    # ── Step 3: Build model ────────────────────────────────────────────────
    print(
        f"\n[3/5] Building MobileNetV2 classifier "
        f"(augmentation={'ON' if use_augmentation else 'OFF'})…"
    )
    model, mobilenet_backbone = build_clothing_classifier_model(
        use_augmentation=use_augmentation
    )
    model.summary()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    total = sum(tf.size(v).numpy() for v in model.variables)
    print(f"      Trainable params : {trainable:,} / {total:,}")

    # ── Step 4: Stage A – Shape learning on Fashion-MNIST ─────────────────
    # Fashion-MNIST is perfectly balanced (6 000/class) so class weights are
    # uniform.  No class_weight needed.
    print(f"\n[4/5] Stage A – Shape learning on Fashion-MNIST "
          f"(up to {args.epochs} epochs total)…")
    print(f"      Training samples : {len(fm_train_images):,}")

    train_model_phase1(
        model, fm_train_images, fm_train_labels, args.epochs,
    )
    train_model_phase2(
        model,
        mobilenet_backbone,
        fm_train_images,
        fm_train_labels,
        args.epochs,
        phase2_lr=args.phase2_lr,
        fine_tune_layers=fine_tune_layers,
    )

    # ── Stage B – Real-photo color adaptation (Kaggle) ────────────────────
    if kaggle_images is not None and kaggle_labels is not None:
        print("\n      Computing Kaggle class weights…")
        kaggle_class_weights = compute_class_weights(kaggle_labels)
        train_model_phase3(
            model,
            mobilenet_backbone,
            kaggle_images,
            kaggle_labels,
            phase3_lr=args.phase3_lr,
            kaggle_epochs=args.kaggle_epochs,
            class_weight=kaggle_class_weights,
        )

    # ── Step 5: Evaluate & export ──────────────────────────────────────────
    print("\n[5/5] Evaluating on Fashion-MNIST test set…")
    evaluate_model_accuracy(model, test_images, test_labels_onehot)

    print("\nExporting to TFLite…")
    export_model_to_tflite(model, args.output)

    print("\nDone! The TFLite model is ready to be bundled in the Flutter app.")


if __name__ == "__main__":
    main()
