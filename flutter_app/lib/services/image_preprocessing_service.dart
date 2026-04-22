/// Service responsible for converting a raw camera image into the exact
/// pixel format expected by the MobileNetV2-based TFLite model.
///
/// Model input specification
/// -------------------------
/// • Size      : [modelInputSize] × [modelInputSize] pixels (96 × 96)
/// • Channels  : 3 (RGB)
/// • Range     : −1.0 … +1.0  (MobileNetV2 standard: pixel / 127.5 − 1.0)
///
/// The Python training script applies the identical three-step pipeline to
/// each Fashion-MNIST image:
///   1. Resize 28×28 → 96×96 (bilinear)
///   2. Repeat grayscale channel 3× → pseudo-RGB
///   3. Normalise: (pixel / 127.5) − 1.0
///
/// This service performs the same pipeline on live camera images so that
/// training and inference are perfectly aligned.
///
/// Extensibility
/// -------------
/// To add color recognition in the future, keep this service unchanged –
/// it already outputs full RGB tensors.  A second model simply reads the same
/// [preprocessImageForMobileNetV2] output and adds color-specific heads.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Converts a captured image file into the tensor input expected by the
/// MobileNetV2 clothing classifier.
class ImagePreprocessingService {
  // ── Model input constants ─────────────────────────────────────────────────

  /// Width and height of the model's square input in pixels.
  /// Must match [MODEL_INPUT_SIZE] in train_fashion_mnist.py.
  static const int modelInputSize = 96;

  /// Number of color channels expected by the model (RGB = 3).
  static const int modelInputChannels = 3;

  /// Total number of float values in one input tensor:
  /// modelInputSize × modelInputSize × modelInputChannels.
  static const int inputTensorLength =
      modelInputSize * modelInputSize * modelInputChannels;

  // ─────────────────────────────────────────────────────────────────────────

  /// Loads [imageFile], resizes it to 96×96, and produces a flat
  /// [Float32List] of 27 648 values (96 × 96 × 3) normalized to [−1.0, 1.0].
  ///
  /// Preprocessing pipeline (identical to the Python training script):
  ///   1. Decode image from disk.
  ///   2. Center-crop to a square so the subject is not distorted.
  ///   3. Resize to [modelInputSize] × [modelInputSize] (bilinear).
  ///   4. For each pixel and channel: value / 127.5 − 1.0  → [−1, 1].
  ///
  /// The returned list is stored in row-major, channel-last order:
  ///   index = row * modelInputSize * 3 + col * 3 + channel
  ///
  /// Throws a [StateError] when the image file cannot be decoded.
  static Float32List preprocessImageForMobileNetV2(File imageFile) {
    // --- Step 1: Decode the image file into an in-memory bitmap ----------
    final Uint8List rawBytes = imageFile.readAsBytesSync();
    final img.Image? decoded = img.decodeImage(rawBytes);

    if (decoded == null) {
      throw StateError(
        'ImagePreprocessingService: failed to decode image at '
        '${imageFile.path}',
      );
    }

    // --- Step 2: Center-crop to a square ---------------------------------
    // Cropping before resizing preserves the aspect ratio and avoids
    // squashing tall or wide images.  The shortest side determines the
    // square size; the longer side is cropped symmetrically.
    final img.Image squareImage = _centerCropToSquare(decoded);

    // --- Step 3: Resize to modelInputSize × modelInputSize ---------------
    final img.Image resizedImage = img.copyResize(
      squareImage,
      width: modelInputSize,
      height: modelInputSize,
      interpolation: img.Interpolation.linear,
    );

    // --- Step 4: Build float32 tensor with MobileNetV2 normalisation -----
    // For each pixel: channel value in [0, 255] → (value / 127.5) − 1.0 ∈ [−1, 1].
    final Float32List tensor = Float32List(inputTensorLength);
    int index = 0;

    for (int row = 0; row < modelInputSize; row++) {
      for (int col = 0; col < modelInputSize; col++) {
        final img.Pixel pixel = resizedImage.getPixel(col, row);
        for (final num channel in [pixel.r, pixel.g, pixel.b]) {
          tensor[index++] = channel / 127.5 - 1.0;
        }
      }
    }

    return tensor;
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /// Crops [source] to the largest centered square it contains.
  static img.Image _centerCropToSquare(img.Image source) {
    final int side = source.width < source.height ? source.width : source.height;
    final int xOffset = (source.width - side) ~/ 2;
    final int yOffset = (source.height - side) ~/ 2;
    return img.copyCrop(
      source,
      x: xOffset,
      y: yOffset,
      width: side,
      height: side,
    );
  }
}
