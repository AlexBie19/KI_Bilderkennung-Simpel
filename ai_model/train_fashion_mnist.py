#!/usr/bin/env python3
"""
train_fashion_mnist.py
======================
Trains a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset and
exports the trained model as a TensorFlow Lite (TFLite) file that can be
bundled inside the Flutter app.

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

Extension notes
---------------
To add color and pattern recognition later:
  1. Collect an RGB dataset (or augment Fashion-MNIST with synthetic color).
  2. Define a new model architecture with 3 input channels.
  3. Add new output heads or create a second model for color / pattern.
  4. Export that model as a separate TFLite file and add a new service class
     in the Flutter app (no changes to the existing ClothingClassifierService).

Usage
-----
  python train_fashion_mnist.py [--epochs N] [--output PATH]

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
DEFAULT_TRAINING_EPOCHS: int = 15
DEFAULT_BATCH_SIZE: int = 64
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
    """Load Fashion-MNIST and normalise pixel values to [0.0, 1.0].

    Returns
    -------
    training_images : np.ndarray
        Shape (60 000, 28, 28, 1), dtype float32.
    training_labels : np.ndarray
        Shape (60 000,), dtype uint8.
    test_images : np.ndarray
        Shape (10 000, 28, 28, 1), dtype float32.
    test_labels : np.ndarray
        Shape (10 000,), dtype uint8.
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
    # TensorFlow Conv2D layers expect a channel axis even for grayscale images.
    training_images = np.expand_dims(training_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    return training_images, training_labels, test_images, test_labels


# ── Model architecture ────────────────────────────────────────────────────────

def build_clothing_classifier_model() -> keras.Model:
    """Build and compile the CNN clothing classifier.

    Architecture overview
    ---------------------
    Input  28×28×1 (grayscale)
      │
      ├─ Conv2D(32, 3×3, relu)  ──▶ BatchNorm ──▶ MaxPool(2×2)
      ├─ Conv2D(64, 3×3, relu)  ──▶ BatchNorm ──▶ MaxPool(2×2)
      ├─ Conv2D(128, 3×3, relu) ──▶ BatchNorm
      │
      ├─ GlobalAveragePooling2D
      ├─ Dense(256, relu) ──▶ Dropout(0.4)
      └─ Dense(10, softmax)  ← one output neuron per clothing class

    Design rationale
    ----------------
    - Global average pooling replaces Flatten + large Dense to reduce the
      number of parameters while retaining spatial feature information.
    - Batch normalisation speeds up convergence and reduces overfitting.
    - Dropout regularises the final fully-connected layer.
    - Softmax output is compatible with both categorical cross-entropy loss
      and the argmax extraction used in ClothingClassifierService.

    Extension note
    --------------
    To support color / pattern recognition, define a new function
    `build_color_pattern_classifier_model()` with an input shape of
    (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and possibly extra output heads.
    This model remains unchanged.

    Returns
    -------
    keras.Model
        Compiled but untrained model.
    """
    # Input layer explicitly named for clarity when inspecting the model.
    input_layer = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, GRAYSCALE_CHANNEL_COUNT),
        name="grayscale_image_input",
    )

    # ── First convolutional block ─────────────────────────────────────────
    convolution_block_1 = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_block1_conv",
    )(input_layer)
    convolution_block_1 = keras.layers.BatchNormalization(
        name="conv_block1_batchnorm"
    )(convolution_block_1)
    convolution_block_1 = keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="conv_block1_maxpool"
    )(convolution_block_1)

    # ── Second convolutional block ────────────────────────────────────────
    convolution_block_2 = keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_block2_conv",
    )(convolution_block_1)
    convolution_block_2 = keras.layers.BatchNormalization(
        name="conv_block2_batchnorm"
    )(convolution_block_2)
    convolution_block_2 = keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="conv_block2_maxpool"
    )(convolution_block_2)

    # ── Third convolutional block (no pooling to preserve spatial info) ───
    convolution_block_3 = keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_block3_conv",
    )(convolution_block_2)
    convolution_block_3 = keras.layers.BatchNormalization(
        name="conv_block3_batchnorm"
    )(convolution_block_3)

    # ── Global average pooling ────────────────────────────────────────────
    # Produces a 128-element feature vector regardless of spatial resolution,
    # making the model robust to slight size variations in input images.
    global_average_pooling = keras.layers.GlobalAveragePooling2D(
        name="global_avg_pool"
    )(convolution_block_3)

    # ── Fully-connected classifier head ───────────────────────────────────
    dense_features = keras.layers.Dense(
        units=256, activation="relu", name="dense_features"
    )(global_average_pooling)
    dropout_regularisation = keras.layers.Dropout(
        rate=0.4, name="dropout_regularisation"
    )(dense_features)

    # Output: one probability per clothing class.
    output_probabilities = keras.layers.Dense(
        units=NUMBER_OF_CLOTHING_CLASSES,
        activation="softmax",
        name="clothing_class_probabilities",
    )(dropout_regularisation)

    # Assemble the full model.
    clothing_classifier_model = keras.Model(
        inputs=input_layer,
        outputs=output_probabilities,
        name="fashion_mnist_clothing_classifier",
    )

    # Compile with Adam optimiser and sparse labels (labels are integers,
    # not one-hot vectors) to keep the data loading code simple.
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
    """Train [clothing_classifier_model] on the Fashion-MNIST training data.

    Parameters
    ----------
    clothing_classifier_model : keras.Model
        The compiled (but untrained) model from build_clothing_classifier_model.
    training_images : np.ndarray
        Preprocessed training images, shape (N, 28, 28, 1).
    training_labels : np.ndarray
        Integer class labels, shape (N,).
    number_of_training_epochs : int
        How many full passes over the training data to perform.

    Returns
    -------
    keras.callbacks.History
        Training history (loss and accuracy per epoch).
    """
    # Reduce the learning rate when validation loss plateaus to squeeze out
    # extra accuracy in later epochs.
    learning_rate_reducer_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,          # Halve the learning rate when triggered.
        patience=3,          # Wait 3 epochs before reducing.
        min_lr=1e-6,
        verbose=1,
    )

    # Stop training early if validation loss stops improving to prevent
    # overfitting and save unnecessary compute time.
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,          # Stop after 5 epochs of no improvement.
        restore_best_weights=True,  # Use the best checkpoint, not the last.
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
    """Convert [trained_model] to TFLite format and save it to disk.

    The converter applies default optimisations (float16 quantisation) which
    reduce file size and improve inference speed on mobile devices while
    preserving acceptable accuracy.

    For maximum accuracy you can disable optimisations:
        converter.optimizations = []

    For maximum performance (at the cost of some accuracy) you can apply
    full integer quantisation – see the TFLite documentation.

    Parameters
    ----------
    trained_model : keras.Model
        The model that has been trained and evaluated.
    tflite_output_path : str
        Absolute or relative path where the .tflite file will be written.
    """
    # Initialise the converter from the Keras model.
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)

    # Apply default size/latency optimisations (recommended for mobile).
    tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Run the conversion.
    tflite_model_bytes = tflite_converter.convert()

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(os.path.abspath(tflite_output_path)), exist_ok=True)

    # Write the binary model file.
    with open(tflite_output_path, "wb") as tflite_output_file:
        tflite_output_file.write(tflite_model_bytes)

    tflite_file_size_kb = os.path.getsize(tflite_output_path) / 1024
    print(
        f"TFLite model saved to: {tflite_output_path} "
        f"({tflite_file_size_kb:.1f} KB)"
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
        Held-out test images.
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
        description="Train a Fashion-MNIST CNN and export it as TFLite."
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_TRAINING_EPOCHS}).",
    )
    argument_parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TFLITE_OUTPUT_PATH,
        help="Output path for the .tflite model file.",
    )
    return argument_parser.parse_args()


def main() -> None:
    """Main training pipeline: load → build → train → evaluate → export."""
    # Parse CLI arguments.
    parsed_arguments = parse_command_line_arguments()

    print("=" * 60)
    print("  Fashion-MNIST Clothing Classifier – Training Pipeline")
    print("=" * 60)

    # --- Step 1: Load and preprocess dataset ----------------------------
    print("\n[1/4] Loading Fashion-MNIST dataset…")
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
    print("\n[2/4] Building model architecture…")
    clothing_classifier_model = build_clothing_classifier_model()
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
