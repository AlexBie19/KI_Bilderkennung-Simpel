#!/usr/bin/env python3
"""
train_fashion_mnist.py
======================
Trains a MobileNetV2-based transfer-learning model on the *full* Fashion-MNIST
dataset (all 70 000 samples, all 10 classes) and exports it as a TensorFlow
Lite (TFLite) file ready to be bundled inside the Flutter app.

Why MobileNetV2?
----------------
MobileNetV2 is pre-trained on ImageNet and already understands rich visual
features (edges, textures, shapes).  By fine-tuning its upper layers on
Fashion-MNIST we get much better generalisation to real camera photos than
training a small CNN from scratch.  MobileNetV2 is also specifically designed
for on-device inference, which makes the exported TFLite model fast and
memory-efficient.

The architecture is deliberately kept extensible: adding a second head for
color or pattern recognition later only requires loading the same MobileNetV2
backbone and connecting a new output branch.

Fashion-MNIST overview
----------------------
- 70 000 grayscale images, each 28 × 28 pixels.
- 10 clothing classes:
    0  T-shirt/top   5  Sandal
    1  Trouser        6  Shirt
    2  Pullover       7  Sneaker
    3  Dress          8  Bag
    4  Coat           9  Ankle boot
- 60 000 training samples / 10 000 test samples.

Input pipeline
--------------
Fashion-MNIST images are 28×28 grayscale.  MobileNetV2 requires at least
96×96 RGB input.  The preprocessing pipeline therefore:
  1. Resizes each 28×28 image to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE (96×96).
  2. Repeats the single grayscale channel three times to form a pseudo-RGB
     image (R = G = B = grey).
  3. Normalises pixels from [0, 255] to [-1.0, 1.0] using MobileNetV2's
     standard preprocess_input function.

The Flutter app's ImagePreprocessingService applies the *same* three steps at
inference time (resize → RGB → [-1, 1]) so training and inference are aligned.

Usage
-----
  python train_fashion_mnist.py [--epochs N] [--output PATH] [--no-augment]

Requirements
------------
  See requirements.txt
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ── Constants ────────────────────────────────────────────────────────────────

# MobileNetV2 minimum input size is 96×96. Using 96 keeps the model small and
# fast while giving MobileNetV2 enough spatial resolution to extract features.
# Change to 128 or 224 if you have the compute budget and need higher accuracy.
MODEL_INPUT_SIZE: int = 96

# Number of input color channels (3 = RGB, matching MobileNetV2 expectations).
# Using RGB instead of grayscale also enables future color recognition.
MODEL_INPUT_CHANNELS: int = 3

# Number of output classes (one per Fashion-MNIST clothing category).
NUMBER_OF_CLOTHING_CLASSES: int = 10

# Default training hyper-parameters – override via command-line flags.
DEFAULT_TRAINING_EPOCHS: int = 30       # Maximum epochs (EarlyStopping kicks in)
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_VALIDATION_SPLIT: float = 0.1  # 10 % of training data for validation

# Maximum number of epochs for phase 1 (head-only training).
# The remainder of DEFAULT_TRAINING_EPOCHS is used for phase 2 fine-tuning.
PHASE1_MAX_EPOCHS: int = 15

# Fine-tuning: unfreeze this many layers from the *top* of the MobileNetV2
# backbone during phase 2.  More layers = higher accuracy but longer training.
FINE_TUNE_LAYER_COUNT: int = 50

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

    The Flutter app's ImagePreprocessingService applies the *same* three steps
    so training and inference are perfectly aligned.

    Returns
    -------
    training_images : np.ndarray   shape (60 000, 96, 96, 3), float32
    training_labels : np.ndarray   shape (60 000,), uint8
    test_images     : np.ndarray   shape (10 000, 96, 96, 3), float32
    test_labels     : np.ndarray   shape (10 000,), uint8
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

    return training_images, training_labels, test_images, test_labels


# ── Data augmentation ─────────────────────────────────────────────────────────

def build_augmentation_pipeline() -> keras.Sequential:
    """On-the-fly data augmentation applied only during training.

    Augmentations simulate the variation in real camera photos: small
    rotations, horizontal mirroring (clothing is horizontally symmetric),
    zoom and brightness/contrast shifts.  Vertical flips are excluded because
    clothing items are always upright.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.06),   # ±21.6°
            keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
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
    Input  96×96×3 (pseudo-RGB, MobileNetV2 normalised)
      │
      ├─ [Optional] Data augmentation pipeline
      ├─ MobileNetV2 backbone (pre-trained on ImageNet, include_top=False)
      │    └─ outputs 3×3×1280 feature map at 96×96 input
      ├─ GlobalAveragePooling2D  →  1280-D feature vector
      ├─ BatchNormalization
      ├─ Dense(256, relu)
      ├─ Dropout(0.4)
      └─ Dense(10, softmax)   ← one neuron per Fashion-MNIST class

    Training strategy
    -----------------
    Phase 1 (head training):
        Freeze the entire MobileNetV2 backbone.  Train only the custom head
        for a few epochs so the new dense layers converge before the backbone
        weights are touched.

    Phase 2 (fine-tuning):
        Unfreeze the top FINE_TUNE_LAYER_COUNT layers of MobileNetV2 and
        continue training with a 10× smaller learning rate.  Lower layers
        retain ImageNet features; upper layers adapt to clothing-specific
        textures and shapes.

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
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dense(units=256, activation="relu", name="dense_features")(x)
    x = keras.layers.Dropout(rate=0.4, name="dropout_regularisation")(x)

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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
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
    training_labels: np.ndarray,
    number_of_epochs: int,
) -> None:
    """Phase 1: train only the custom head (backbone is frozen).

    Parameters
    ----------
    model            : keras.Model   Compiled model from build_clothing_classifier_model.
    training_images  : np.ndarray    Preprocessed images, shape (N, 96, 96, 3).
    training_labels  : np.ndarray    Integer class labels, shape (N,).
    number_of_epochs : int           Upper bound on training epochs.
    """
    # Limit phase 1 to at most PHASE1_MAX_EPOCHS (EarlyStopping will cut it shorter).
    phase1_epochs = min(number_of_epochs, PHASE1_MAX_EPOCHS)

    print(f"\n  Phase 1 – training head only (up to {phase1_epochs} epochs)…")
    model.fit(
        training_images,
        training_labels,
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
    training_labels: np.ndarray,
    number_of_epochs: int,
) -> None:
    """Phase 2: unfreeze the top layers of MobileNetV2 and fine-tune.

    The top FINE_TUNE_LAYER_COUNT layers are unfrozen; lower layers remain
    frozen so their low-level ImageNet features are preserved.  A 10× smaller
    learning rate prevents large gradient updates from destroying the pre-
    trained weights.

    Parameters
    ----------
    model               : keras.Model   The full model (head + backbone).
    mobilenet_backbone  : keras.Model   The MobileNetV2 sub-model to partially unfreeze.
    training_images     : np.ndarray    Same preprocessed images as phase 1.
    training_labels     : np.ndarray    Same labels as phase 1.
    number_of_epochs    : int           Upper bound on fine-tuning epochs.
    """
    # Unfreeze everything, then re-freeze the lower layers.
    mobilenet_backbone.trainable = True
    total_layers = len(mobilenet_backbone.layers)
    freeze_until = total_layers - FINE_TUNE_LAYER_COUNT
    for layer in mobilenet_backbone.layers[:freeze_until]:
        layer.trainable = False

    frozen_count = sum(1 for l in mobilenet_backbone.layers if not l.trainable)
    print(
        f"\n  Phase 2 – fine-tuning top {FINE_TUNE_LAYER_COUNT} of "
        f"{total_layers} backbone layers "
        f"({frozen_count} layers still frozen)…"
    )

    # Recompile with a much smaller LR so the backbone weights change slowly.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Remaining epochs for fine-tuning.
    remaining_epochs = max(number_of_epochs - PHASE1_MAX_EPOCHS, 10)
    model.fit(
        training_images,
        training_labels,
        epochs=remaining_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        validation_split=DEFAULT_VALIDATION_SPLIT,
        callbacks=_make_callbacks(patience_reduce=4, patience_stop=8),
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
    test_labels: np.ndarray,
) -> None:
    """Print the test-set loss and accuracy of [trained_model]."""
    test_loss, test_accuracy = trained_model.evaluate(
        test_images, test_labels, verbose=0
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
    return parser.parse_args()


def main() -> None:
    """Main pipeline: load → build → phase-1 train → phase-2 fine-tune → evaluate → export."""
    args = parse_command_line_arguments()
    use_augmentation: bool = not args.no_augment

    print("=" * 65)
    print("  Fashion-MNIST Clothing Classifier – MobileNetV2 Pipeline")
    print("=" * 65)
    print(f"  Input size  : {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE}×{MODEL_INPUT_CHANNELS}")
    print(f"  Augmentation: {'ON' if use_augmentation else 'OFF'}")
    print(f"  Max epochs  : {args.epochs} (phase 1 ≤15, phase 2 remainder)")

    # --- Step 1: Load dataset ------------------------------------------
    print("\n[1/4] Loading & preprocessing Fashion-MNIST (all 70 000 samples)…")
    training_images, training_labels, test_images, test_labels = (
        load_and_preprocess_fashion_mnist_dataset()
    )
    print(f"      Training samples : {len(training_images)}")
    print(f"      Test samples     : {len(test_images)}")
    print(f"      Image shape      : {training_images.shape[1:]}")

    # --- Step 2: Build model ------------------------------------------
    print(
        f"\n[2/4] Building MobileNetV2 classifier "
        f"(augmentation={'ON' if use_augmentation else 'OFF'})…"
    )
    model, mobilenet_backbone = build_clothing_classifier_model(
        use_augmentation=use_augmentation
    )
    model.summary()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    total = sum(tf.size(v).numpy() for v in model.variables)
    print(f"      Trainable params : {trainable:,} / {total:,}")

    # --- Step 3: Two-phase training ------------------------------------
    print(f"\n[3/4] Training (two phases, up to {args.epochs} epochs total)…")
    train_model_phase1(model, training_images, training_labels, args.epochs)
    train_model_phase2(model, mobilenet_backbone, training_images, training_labels, args.epochs)

    # --- Step 4: Evaluate & export ------------------------------------
    print("\n[4/4] Evaluating on test set…")
    evaluate_model_accuracy(model, test_images, test_labels)

    print("\nExporting to TFLite…")
    export_model_to_tflite(model, args.output)

    print("\nDone! The TFLite model is ready to be bundled in the Flutter app.")


if __name__ == "__main__":
    main()
