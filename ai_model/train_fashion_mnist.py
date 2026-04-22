#!/usr/bin/env python3
"""
train_fashion_mnist.py
======================
Trains a deep Convolutional Neural Network (CNN) on the *full* Fashion-MNIST
dataset (all 70 000 samples, all 10 classes) and exports the trained model as
a TensorFlow Lite (TFLite) file ready to be bundled inside the Flutter app.

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

Architecture
------------
A deeper residual CNN with four convolutional blocks, batch normalisation,
skip connections, global average pooling and two dense layers.  Trained with
aggressive data augmentation (random flips, rotations, zoom, brightness and
contrast jitter) so that the exported model generalises well to real camera
photos that are pre-processed to 28 × 28 grayscale.

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


# ── Constants ────────────────────────────────────────────────────────────────

# Image dimensions used by Fashion-MNIST (do not change without retraining).
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
GRAYSCALE_CHANNEL_COUNT: int = 1  # Fashion-MNIST is single-channel

# Number of output classes (one per clothing category).
NUMBER_OF_CLOTHING_CLASSES: int = 10

# Default training hyper-parameters – override via command-line flags.
DEFAULT_TRAINING_EPOCHS: int = 30
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_VALIDATION_SPLIT: float = 0.1  # 10 % of training data used for val.

# Where to write the finished TFLite model file.
DEFAULT_TFLITE_OUTPUT_PATH: str = os.path.join(
    os.path.dirname(__file__),  # same folder as this script
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
    """Load the *full* Fashion-MNIST dataset and normalise pixel values to [0, 1].

    All 60 000 training samples and all 10 000 test samples are used.

    Returns
    -------
    training_images : np.ndarray   shape (60 000, 28, 28, 1), dtype float32
    training_labels : np.ndarray   shape (60 000,), dtype uint8
    test_images     : np.ndarray   shape (10 000, 28, 28, 1), dtype float32
    test_labels     : np.ndarray   shape (10 000,), dtype uint8
    """
    # Keras downloads and caches the dataset automatically.
    (raw_training_images, training_labels), (raw_test_images, test_labels) = (
        keras.datasets.fashion_mnist.load_data()
    )

    # --- Normalise: uint8 [0, 255] → float32 [0.0, 1.0] ---
    # The model expects the same normalisation applied to camera images in
    # ImagePreprocessingService.preprocessImageForFashionMnist().
    training_images = raw_training_images.astype("float32") / 255.0
    test_images = raw_test_images.astype("float32") / 255.0

    # --- Add channel dimension: (N, 28, 28) → (N, 28, 28, 1) ---
    # TensorFlow Conv2D layers require a channel axis even for grayscale images.
    training_images = np.expand_dims(training_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    return training_images, training_labels, test_images, test_labels


# ── Data augmentation ─────────────────────────────────────────────────────────

def build_augmentation_pipeline() -> keras.Sequential:
    """Build an on-the-fly data augmentation pipeline applied during training.

    Augmentations are chosen to simulate the variation seen in real camera
    photos: slight rotations, horizontal mirroring, zoom and brightness shifts.
    They do *not* include vertical flips because clothing photos are always
    upright.

    Returns a Keras Sequential model that can be passed as a ``preprocessing``
    argument or called as a layer inside the model graph.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.08),   # ±~29 °
            keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            keras.layers.RandomBrightness(factor=0.15),
            keras.layers.RandomContrast(factor=0.15),
        ],
        name="data_augmentation",
    )


# ── Residual block helper ─────────────────────────────────────────────────────

def _residual_block(
    input_tensor: tf.Tensor,
    filters: int,
    block_name: str,
    stride: int = 1,
) -> tf.Tensor:
    """Two-layer residual block with optional projection shortcut.

    Layout: Conv → BN → ReLU → Conv → BN → (add shortcut) → ReLU

    Parameters
    ----------
    input_tensor : tf.Tensor   Input feature map.
    filters      : int         Number of output filters for both Conv layers.
    block_name   : str         Unique prefix for all layer names in this block.
    stride       : int         Stride applied to the first Conv (for downscaling).

    Returns
    -------
    tf.Tensor   Output feature map after the residual addition.
    """
    # ── Main path ─────────────────────────────────────────────────────────
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv1",
    )(input_tensor)
    x = keras.layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = keras.layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv2",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{block_name}_bn2")(x)

    # ── Shortcut path ─────────────────────────────────────────────────────
    # A 1×1 projection is needed when the spatial size or filter count changes.
    shortcut = input_tensor
    input_filters = input_tensor.shape[-1]
    if stride != 1 or input_filters != filters:
        shortcut = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding="same",
            use_bias=False,
            name=f"{block_name}_shortcut_conv",
        )(input_tensor)
        shortcut = keras.layers.BatchNormalization(
            name=f"{block_name}_shortcut_bn"
        )(shortcut)

    x = keras.layers.Add(name=f"{block_name}_add")([x, shortcut])
    x = keras.layers.Activation("relu", name=f"{block_name}_relu2")(x)
    return x


# ── Model architecture ────────────────────────────────────────────────────────

def build_clothing_classifier_model(use_augmentation: bool = True) -> keras.Model:
    """Build and compile a deep residual CNN clothing classifier.

    Architecture overview
    ---------------------
    Input  28×28×1 (grayscale)
      │
      ├─ [Optional] Data augmentation pipeline
      ├─ Conv2D(32, 3×3) → BN → ReLU                (entry stem)
      │
      ├─ ResidualBlock(32)  × 2
      ├─ ResidualBlock(64,  stride=2)  × 2           (14×14)
      ├─ ResidualBlock(128, stride=2)  × 2           (7×7)
      ├─ ResidualBlock(256, stride=2)  × 2           (4×4)
      │
      ├─ GlobalAveragePooling2D                      (256-D feature vector)
      ├─ Dense(256, relu) → BatchNorm → Dropout(0.4)
      └─ Dense(10, softmax)   ← one neuron per Fashion-MNIST class

    Returns
    -------
    keras.Model
        Compiled but untrained model.
    """
    input_layer = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, GRAYSCALE_CHANNEL_COUNT),
        name="grayscale_image_input",
    )

    x = input_layer

    # ── Optional data augmentation (active only during training) ─────────
    if use_augmentation:
        x = build_augmentation_pipeline()(x)

    # ── Entry stem ────────────────────────────────────────────────────────
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.Activation("relu", name="stem_relu")(x)

    # ── Stage 1: 28×28 → 28×28, 32 filters ───────────────────────────────
    x = _residual_block(x, filters=32, block_name="stage1_block1")
    x = _residual_block(x, filters=32, block_name="stage1_block2")

    # ── Stage 2: 28×28 → 14×14, 64 filters ───────────────────────────────
    x = _residual_block(x, filters=64, block_name="stage2_block1", stride=2)
    x = _residual_block(x, filters=64, block_name="stage2_block2")

    # ── Stage 3: 14×14 → 7×7, 128 filters ────────────────────────────────
    x = _residual_block(x, filters=128, block_name="stage3_block1", stride=2)
    x = _residual_block(x, filters=128, block_name="stage3_block2")

    # ── Stage 4: 7×7 → 4×4, 256 filters ─────────────────────────────────
    x = _residual_block(x, filters=256, block_name="stage4_block1", stride=2)
    x = _residual_block(x, filters=256, block_name="stage4_block2")

    # ── Global average pooling → 256-D feature vector ────────────────────
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ── Classifier head ───────────────────────────────────────────────────
    x = keras.layers.Dense(units=256, use_bias=False, name="dense_features")(x)
    x = keras.layers.BatchNormalization(name="dense_bn")(x)
    x = keras.layers.Activation("relu", name="dense_relu")(x)
    x = keras.layers.Dropout(rate=0.4, name="dropout_regularisation")(x)

    output_probabilities = keras.layers.Dense(
        units=NUMBER_OF_CLOTHING_CLASSES,
        activation="softmax",
        name="clothing_class_probabilities",
    )(x)

    clothing_classifier_model = keras.Model(
        inputs=input_layer,
        outputs=output_probabilities,
        name="fashion_mnist_clothing_classifier",
    )

    clothing_classifier_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return clothing_classifier_model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    clothing_classifier_model: keras.Model,
    training_images: np.ndarray,
    training_labels: np.ndarray,
    number_of_training_epochs: int,
) -> keras.callbacks.History:
    """Train [clothing_classifier_model] on the full Fashion-MNIST training set.

    Parameters
    ----------
    clothing_classifier_model : keras.Model
        The compiled (but untrained) model from build_clothing_classifier_model.
    training_images : np.ndarray
        Preprocessed training images, shape (N, 28, 28, 1).
    training_labels : np.ndarray
        Integer class labels, shape (N,).
    number_of_training_epochs : int
        Maximum number of full passes over the training data.

    Returns
    -------
    keras.callbacks.History
        Training history (loss and accuracy per epoch).
    """
    # Reduce the learning rate when validation loss plateaus.
    learning_rate_reducer_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,       # Halve the learning rate when triggered.
        patience=4,       # Wait 4 epochs before reducing.
        min_lr=1e-6,
        verbose=1,
    )

    # Stop training early if validation loss stops improving.
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,                  # Stop after 8 epochs of no improvement.
        restore_best_weights=True,   # Use the best checkpoint, not the last.
        verbose=1,
    )

    training_history = clothing_classifier_model.fit(
        training_images,
        training_labels,
        epochs=number_of_training_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        validation_split=DEFAULT_VALIDATION_SPLIT,
        callbacks=[learning_rate_reducer_callback, early_stopping_callback],
        verbose=1,
    )

    return training_history


# ── TFLite export ─────────────────────────────────────────────────────────────

def export_model_to_tflite(
    trained_model: keras.Model, tflite_output_path: str
) -> None:
    """Convert [trained_model] to TFLite float32 format and save to disk.

    The model is exported with float32 weights and activations so that the
    Flutter app can feed it float32 tensors directly without any quantisation
    mismatch.  Dynamic range quantisation (DEFAULT) is applied to reduce the
    file size while keeping float32 I/O compatibility.

    Parameters
    ----------
    trained_model : keras.Model
        The model that has been trained and evaluated.
    tflite_output_path : str
        Absolute or relative path where the .tflite file will be written.
    """
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)

    # Dynamic range quantisation compresses weights to int8 but keeps the
    # activations (and model I/O) as float32, so the Flutter app feeds
    # float32 tensors and receives float32 output probabilities unchanged.
    tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model_bytes = tflite_converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(tflite_output_path)), exist_ok=True)

    with open(tflite_output_path, "wb") as tflite_output_file:
        tflite_output_file.write(tflite_model_bytes)

    tflite_file_size_kb = os.path.getsize(tflite_output_path) / 1024
    print(
        f"TFLite model saved → {tflite_output_path} "
        f"({tflite_file_size_kb:.1f} KB)"
    )

    # --- Quick sanity check: run one inference with a blank image ---------
    interpreter = tf.lite.Interpreter(model_path=tflite_output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.zeros(
        input_details[0]["shape"], dtype=np.float32
    )
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]["index"])

    print(
        f"Sanity check passed – output shape: {test_output.shape}, "
        f"sum of probabilities: {test_output.sum():.4f}"
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model_accuracy(
    trained_model: keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    """Print the test-set loss and accuracy of [trained_model].

    Parameters
    ----------
    trained_model : keras.Model
        Fully trained Keras model.
    test_images : np.ndarray
        Held-out test images (10 000 samples).
    test_labels : np.ndarray
        Corresponding integer class labels.
    """
    test_loss, test_accuracy = trained_model.evaluate(
        test_images, test_labels, verbose=0
    )
    print(f"Test accuracy : {test_accuracy * 100:.2f} %")
    print(f"Test loss     : {test_loss:.4f}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command-line flags and return the argument namespace."""
    argument_parser = argparse.ArgumentParser(
        description=(
            "Train a deep residual CNN on Fashion-MNIST and export it as TFLite."
        )
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS,
        help=f"Maximum training epochs (default: {DEFAULT_TRAINING_EPOCHS}).",
    )
    argument_parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TFLITE_OUTPUT_PATH,
        help="Output path for the .tflite model file.",
    )
    argument_parser.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="Disable data augmentation (faster training, lower accuracy).",
    )
    return argument_parser.parse_args()


def main() -> None:
    """Main training pipeline: load → build → train → evaluate → export."""
    parsed_arguments = parse_command_line_arguments()
    use_augmentation: bool = not parsed_arguments.no_augment

    print("=" * 65)
    print("  Fashion-MNIST Clothing Classifier – Training Pipeline")
    print("=" * 65)

    # --- Step 1: Load and preprocess dataset ----------------------------
    print("\n[1/4] Loading Fashion-MNIST dataset (all 70 000 samples)…")
    (
        training_images,
        training_labels,
        test_images,
        test_labels,
    ) = load_and_preprocess_fashion_mnist_dataset()
    print(f"      Training samples : {len(training_images)}")
    print(f"      Test samples     : {len(test_images)}")
    print(f"      Image shape      : {training_images.shape[1:]}")

    # --- Step 2: Build model -------------------------------------------
    print(
        f"\n[2/4] Building model architecture "
        f"(augmentation={'ON' if use_augmentation else 'OFF'})…"
    )
    clothing_classifier_model = build_clothing_classifier_model(
        use_augmentation=use_augmentation
    )
    clothing_classifier_model.summary()

    # --- Step 3: Train -------------------------------------------------
    print(f"\n[3/4] Training for up to {parsed_arguments.epochs} epochs…")
    train_model(
        clothing_classifier_model=clothing_classifier_model,
        training_images=training_images,
        training_labels=training_labels,
        number_of_training_epochs=parsed_arguments.epochs,
    )

    # --- Step 4: Evaluate & export ------------------------------------
    print("\n[4/4] Evaluating on test set…")
    evaluate_model_accuracy(
        trained_model=clothing_classifier_model,
        test_images=test_images,
        test_labels=test_labels,
    )

    print("\nExporting to TFLite…")
    export_model_to_tflite(
        trained_model=clothing_classifier_model,
        tflite_output_path=parsed_arguments.output,
    )

    print("\nDone! The TFLite model is ready to be bundled in the Flutter app.")


if __name__ == "__main__":
    main()
