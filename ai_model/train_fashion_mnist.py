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
Phase 2 (backbone top-100 unfrozen):   at least 30 epochs with LR=5e-5.

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
# backbone during phase 2.  Increasing from 50 to 100 lets more of the
# backbone adapt to clothing-specific textures while the lower layers (which
# capture generic edges and gradients) remain frozen and stable.
FINE_TUNE_LAYER_COUNT: int = 100

# Learning rate used during Phase 2 fine-tuning.
# 5e-5 is intentionally 20× smaller than the Phase-1 rate (1e-3) so the
# pre-trained backbone weights change slowly and are not destroyed.
DEFAULT_PHASE2_LR: float = 5e-5

# Label-smoothing coefficient applied to the cross-entropy loss.
# A value of 0.1 shifts the target probabilities from hard 0/1 to 0.01/0.91,
# which prevents the model from becoming overconfident on easy training
# examples and generally improves generalisation and calibration.
LABEL_SMOOTHING: float = 0.1

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
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and preprocess the Kaggle Fashion Product Images (Small) dataset.

    The function reads ``styles.csv``, filters rows whose ``articleType`` is
    in KAGGLE_ARTICLE_TYPE_TO_CLASS, loads the corresponding JPEG images, and
    applies the same preprocessing pipeline as Fashion-MNIST (resize → RGB →
    MobileNetV2 normalisation) so both datasets are compatible.

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
        If set, randomly sample at most this many images from the filtered
        dataset.  Useful to cap memory usage (default: None = use all).

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

    # Optionally cap the number of samples (random shuffle before cap so each
    # run picks a different subset if the user reruns training).
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    print(f"      Found {len(df):,} usable rows after filtering "
          f"(max_samples={max_samples}).")
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


# ── Dataset combination ───────────────────────────────────────────────────────

def combine_datasets(
    images_a: np.ndarray,
    labels_a: np.ndarray,
    images_b: np.ndarray,
    labels_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate two preprocessed datasets and shuffle them uniformly.

    Both datasets must already be preprocessed to the same spatial size and
    normalisation (i.e. output of load_and_preprocess_fashion_mnist_dataset
    and load_and_preprocess_fashion_product_dataset).

    Parameters
    ----------
    images_a, labels_a : np.ndarray  First dataset (e.g. Fashion-MNIST train).
    images_b, labels_b : np.ndarray  Second dataset (e.g. Kaggle train).

    Returns
    -------
    (combined_images, combined_labels) : tuple[np.ndarray, np.ndarray]
        Shuffled concatenation with shape
        (N_a + N_b, 128, 128, 3) and (N_a + N_b, 10).
    """
    combined_images = np.concatenate([images_a, images_b], axis=0)
    combined_labels = np.concatenate([labels_a, labels_b], axis=0)

    # Fixed seed so the shuffle is reproducible across runs.
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(combined_images))
    return combined_images[indices], combined_labels[indices]


# ── Data augmentation ─────────────────────────────────────────────────────────

def build_augmentation_pipeline() -> keras.Sequential:
    """On-the-fly data augmentation applied only during training.

    Augmentations simulate the variation in real camera photos: small
    rotations, horizontal mirroring (clothing is horizontally symmetric),
    zoom, brightness/contrast shifts, and small translations to simulate
    the garment not being perfectly centred in the frame.  Vertical flips
    are excluded because clothing items are always upright.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.08),       # ±28.8°
            keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            keras.layers.RandomTranslation(
                height_factor=0.08, width_factor=0.08
            ),
            keras.layers.RandomBrightness(factor=0.10),
            keras.layers.RandomContrast(factor=0.10),
        ],
        name="data_augmentation",
    )


# ── Model architecture ────────────────────────────────────────────────────────

def build_clothing_classifier_model(use_augmentation: bool = True) -> keras.Model:
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

    Label smoothing (0.1) is applied via CategoricalCrossentropy so the model
    is penalised for being overconfident, improving generalisation.

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
    # CategoricalCrossentropy with label_smoothing=0.1 replaces the previous
    # SparseCategoricalCrossentropy; labels are now one-hot encoded so that
    # the smoothing parameter is respected.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    return model, mobilenet_backbone


# ── Training ──────────────────────────────────────────────────────────────────

def _make_callbacks(
    patience_reduce: int = 3,
    patience_stop: int = 6,
) -> list:
    """Return standard ReduceLROnPlateau + EarlyStopping callbacks."""
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_reduce,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
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
) -> None:
    """Phase 1: train only the custom head (backbone is frozen).

    Parameters
    ----------
    model                   : keras.Model   Compiled model from build_clothing_classifier_model.
    training_images         : np.ndarray    Preprocessed images, shape (N, 128, 128, 3).
    training_labels_onehot  : np.ndarray    One-hot labels, shape (N, 10).
    number_of_epochs        : int           Upper bound on training epochs.
    """
    # Limit phase 1 to at most PHASE1_MAX_EPOCHS (EarlyStopping will cut it shorter).
    phase1_epochs = min(number_of_epochs, PHASE1_MAX_EPOCHS)

    print(f"\n  Phase 1 – training head only (up to {phase1_epochs} epochs)…")
    model.fit(
        training_images,
        training_labels_onehot,
        epochs=phase1_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        validation_split=DEFAULT_VALIDATION_SPLIT,
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
) -> None:
    """Phase 2: unfreeze the top layers of MobileNetV2 and fine-tune.

    The top fine_tune_layers layers are unfrozen; lower layers remain
    frozen so their low-level ImageNet features are preserved.  A very small
    learning rate (default 5e-5, 200× smaller than Phase-1) prevents large
    gradient updates from destroying the pre-trained weights.

    Parameters
    ----------
    model                   : keras.Model   The full model (head + backbone).
    mobilenet_backbone      : keras.Model   The MobileNetV2 sub-model to partially unfreeze.
    training_images         : np.ndarray    Same preprocessed images as phase 1.
    training_labels_onehot  : np.ndarray    Same one-hot labels as phase 1.
    number_of_epochs        : int           Upper bound on total epochs (both phases).
    phase2_lr               : float         Learning rate for fine-tuning (default 5e-5).
    fine_tune_layers        : int           Number of top backbone layers to unfreeze.
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
    # Label smoothing is kept identical to phase 1 for consistent training.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase2_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    # Guarantee at least 30 fine-tuning epochs (EarlyStopping provides the
    # safety net if the model stops improving).
    remaining_epochs = max(number_of_epochs - PHASE1_MAX_EPOCHS, 30)
    model.fit(
        training_images,
        training_labels_onehot,
        epochs=remaining_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        validation_split=DEFAULT_VALIDATION_SPLIT,
        callbacks=_make_callbacks(patience_reduce=5, patience_stop=10),
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
    return parser.parse_args()


def main() -> None:
    """Main pipeline: load → build → phase-1 train → phase-2 fine-tune → evaluate → export."""
    args = parse_command_line_arguments()
    use_augmentation: bool = not args.no_augment
    fine_tune_layers: int = args.fine_tune_layers

    print("=" * 65)
    print("  Clothing Classifier – MobileNetV2 Dual-Dataset Pipeline")
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
    else:
        print("  Kaggle dataset   : not used (pass --product-dataset-dir to enable)")

    # ── Step 1: Load Fashion-MNIST ─────────────────────────────────────────
    print("\n[1/5] Loading & preprocessing Fashion-MNIST (all 70 000 samples)…")
    fm_train_images, fm_train_labels, test_images, test_labels_onehot = (
        load_and_preprocess_fashion_mnist_dataset()
    )
    print(f"      Fashion-MNIST training : {len(fm_train_images):,} samples")
    print(f"      Fashion-MNIST test     : {len(test_images):,} samples")

    # ── Step 2: Optionally load Kaggle color photos ────────────────────────
    training_images = fm_train_images
    training_labels_onehot = fm_train_labels

    if args.product_dataset_dir is not None:
        print(
            f"\n[2/5] Loading Kaggle Fashion Product Images "
            f"(max {args.max_product_samples:,} samples)…"
        )
        kaggle_result = load_and_preprocess_fashion_product_dataset(
            dataset_dir=args.product_dataset_dir,
            max_samples=args.max_product_samples,
        )
        if kaggle_result is not None:
            kaggle_images, kaggle_labels = kaggle_result
            print(f"      Kaggle samples loaded  : {len(kaggle_images):,}")
            print("\n      Combining Fashion-MNIST + Kaggle datasets…")
            training_images, training_labels_onehot = combine_datasets(
                fm_train_images, fm_train_labels,
                kaggle_images, kaggle_labels,
            )
            print(f"      Combined training set  : {len(training_images):,} samples")
        else:
            print("      Kaggle dataset skipped – training on Fashion-MNIST only.")
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

    # ── Step 4: Two-phase training ─────────────────────────────────────────
    print(f"\n[4/5] Training (two phases, up to {args.epochs} epochs total)…")
    print(f"      Total training samples : {len(training_images):,}")
    train_model_phase1(model, training_images, training_labels_onehot, args.epochs)
    train_model_phase2(
        model,
        mobilenet_backbone,
        training_images,
        training_labels_onehot,
        args.epochs,
        phase2_lr=args.phase2_lr,
        fine_tune_layers=fine_tune_layers,
    )

    # ── Step 5: Evaluate & export ──────────────────────────────────────────
    print("\n[5/5] Evaluating on Fashion-MNIST test set…")
    evaluate_model_accuracy(model, test_images, test_labels_onehot)

    print("\nExporting to TFLite…")
    export_model_to_tflite(model, args.output)

    print("\nDone! The TFLite model is ready to be bundled in the Flutter app.")


if __name__ == "__main__":
    main()
