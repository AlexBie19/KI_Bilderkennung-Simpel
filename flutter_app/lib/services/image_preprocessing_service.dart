/// Service responsible for converting a raw camera image into the exact
/// pixel format expected by the Fashion-MNIST TFLite model.
///
/// Fashion-MNIST uses:
///   • 28 × 28 pixel images
///   • Single-channel (grayscale)
///   • Pixel values normalised to [0.0, 1.0]  (original uint8 ÷ 255)
///
/// This class is intentionally kept stateless so it can be reused without
/// instantiation or disposal.  All public methods are static.
///
/// Extension note: to support color/pattern recognition in the future,
/// add a new method `preprocessForColorModel` that keeps all three RGB
/// channels and resizes to the target model's resolution.

import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Converts a captured image file into the tensor input expected by the
/// Fashion-MNIST clothing classifier.
class ImagePreprocessingService {
  // ── Fashion-MNIST constants ───────────────────────────────────────────────

  /// Width (and height) of every Fashion-MNIST input image in pixels.
  static const int fashionMnistImageSize = 28;

  /// Number of color channels in a Fashion-MNIST image (1 = grayscale).
  static const int fashionMnistChannelCount = 1;

  // ─────────────────────────────────────────────────────────────────────────

  /// Loads [imageFile], converts it to 28 × 28 grayscale, normalises each
  /// pixel value to [0.0, 1.0], and returns a flat [Float32List] that can
  /// be fed directly into the TFLite model's input tensor.
  ///
  /// The returned list contains exactly 28 × 28 = 784 float values in
  /// row-major order (top-left → bottom-right).
  ///
  /// Throws a [StateError] when the image file cannot be decoded.
  static Float32List preprocessImageForFashionMnist(File imageFile) {
    // --- Step 1: Decode the image file into an in-memory bitmap ----------
    final Uint8List rawImageBytes = imageFile.readAsBytesSync();
    final img.Image? decodedImage = img.decodeImage(rawImageBytes);

    if (decodedImage == null) {
      throw StateError(
        'ImagePreprocessingService: Failed to decode image at '
        '${imageFile.path}',
      );
    }

    // --- Step 2: Resize to 28 × 28 using bilinear interpolation ----------
    // Bilinear interpolation preserves fine texture details better than
    // nearest-neighbour when downscaling to very small sizes.
    final img.Image resizedImage = img.copyResize(
      decodedImage,
      width: fashionMnistImageSize,
      height: fashionMnistImageSize,
      interpolation: img.Interpolation.linear,
    );

    // --- Step 3: Convert to grayscale ------------------------------------
    // The `grayscale` function uses the luminosity formula:
    //   Y = 0.299 R + 0.587 G + 0.114 B
    // which matches the perceptual weighting used in Fashion-MNIST.
    final img.Image grayscaleImage = img.grayscale(resizedImage);

    // --- Step 4: Normalise pixels to [0.0, 1.0] and build tensor ---------
    // Fashion-MNIST stores pixel values as uint8 in [0, 255]; dividing by
    // 255 normalises them to the float range expected by the model.
    final Float32List inputTensor = Float32List(
      fashionMnistImageSize * fashionMnistImageSize,
    );

    for (int rowIndex = 0; rowIndex < fashionMnistImageSize; rowIndex++) {
      for (int colIndex = 0; colIndex < fashionMnistImageSize; colIndex++) {
        // img.Pixel exposes the red channel for grayscale images.
        final img.Pixel pixel = grayscaleImage.getPixel(colIndex, rowIndex);
        final double normalisedPixelValue = pixel.r / 255.0;

        // Store in row-major order: row * width + column
        inputTensor[rowIndex * fashionMnistImageSize + colIndex] =
            normalisedPixelValue;
      }
    }

    return inputTensor;
  }

  // ── Future extension ─────────────────────────────────────────────────────
  //
  // /// Preprocesses [imageFile] for a future color-aware model.
  // ///
  // /// Returns an RGB tensor of shape [height, width, 3] normalised to [0, 1].
  // static Float32List preprocessImageForColorModel(
  //   File imageFile, {
  //   required int targetWidth,
  //   required int targetHeight,
  // }) { ... }
  //
  // ─────────────────────────────────────────────────────────────────────────
}
