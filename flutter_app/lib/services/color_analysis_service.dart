/// Service that detects the dominant color of a clothing item from a photo.
///
/// How it works
/// ------------
/// Instead of training a separate color-recognition neural network (which
/// would require a labeled color dataset), this service uses a deterministic
/// HSV (Hue-Saturation-Value) algorithm that runs entirely on the device
/// without any model inference:
///
///   1. Decode the captured image.
///   2. Sample pixels from the **center 60 %** of the image – the clothing
///      item is usually in the middle of the frame, so this avoids background
///      pixels at the edges.
///   3. Convert each sampled RGB pixel to HSV color space.
///   4. Classify the HSV triplet into one of 12 named colors using thresholds
///      (black, white, gray, red, orange, yellow, green, cyan, blue, purple,
///      pink, brown/beige).
///   5. Return the most frequently counted color together with a simple
///      confidence score (fraction of sampled pixels that matched it).
///
/// Why HSV?
/// --------
/// HSV separates chromatic information (hue) from brightness (value) and
/// colorfulness (saturation), which makes color naming very robust to
/// lighting changes.  A red shirt in bright sunlight and in dim indoor light
/// have very different RGB values but a similar hue channel.
///
/// Limitations
/// -----------
/// • Fashion-MNIST is grayscale, so this detector is designed for *real
///   camera photos* taken by the Flutter app, not for the training images.
/// • The algorithm reports the *most common* color, so a striped garment
///   returns the dominant stripe color.
/// • Subtle fashion shades (navy, cobalt, ecru, …) are reported as their
///   nearest basic color (blue, blue, beige).
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

/// Represents one detected dominant color.
class DetectedColor {
  /// Human-readable color name (e.g. "Blue", "Red", "Black").
  final String name;

  /// Fraction of sampled pixels that matched this color [0.0, 1.0].
  final double confidence;

  /// A representative sRGB color for displaying a color swatch in the UI.
  final int red;
  final int green;
  final int blue;

  const DetectedColor({
    required this.name,
    required this.confidence,
    required this.red,
    required this.green,
    required this.blue,
  });

  /// Formats the confidence as a percentage string, e.g. "72.3 %".
  String get formattedConfidence => '${(confidence * 100).toStringAsFixed(1)} %';

  @override
  String toString() =>
      'DetectedColor(name: $name, confidence: ${formattedConfidence})';
}

/// Analyzes the dominant color of clothing in a photograph.
class ColorAnalysisService {
  // ── Sampling parameters ────────────────────────────────────────────────────

  /// Only pixels inside the center fraction of the image are sampled.
  /// 0.6 means the central 60 % in each dimension (i.e. 36 % of all pixels).
  static const double _centerFraction = 0.6;

  /// Sample every N-th pixel to keep analysis fast on large camera images.
  static const int _sampleStride = 4;

  // ─────────────────────────────────────────────────────────────────────────

  /// Detects the dominant color in [imageFile].
  ///
  /// Returns a [DetectedColor] with the most common color name, its
  /// confidence, and a representative RGB swatch.
  ///
  /// Throws a [StateError] if the image cannot be decoded.
  static DetectedColor analyzeColor(File imageFile) {
    final img.Image? decoded =
        img.decodeImage(imageFile.readAsBytesSync());

    if (decoded == null) {
      throw StateError(
        'ColorAnalysisService: failed to decode image at ${imageFile.path}',
      );
    }

    // Determine the central sampling window.
    final int marginX =
        ((decoded.width * (1.0 - _centerFraction)) / 2).round();
    final int marginY =
        ((decoded.height * (1.0 - _centerFraction)) / 2).round();
    final int startX = marginX;
    final int startY = marginY;
    final int endX = decoded.width - marginX;
    final int endY = decoded.height - marginY;

    // Count votes for each color name.
    final Map<String, int> votes = {};
    // Accumulate representative RGB for the winning color.
    final Map<String, List<int>> rgbAccum = {};
    int totalSamples = 0;

    for (int y = startY; y < endY; y += _sampleStride) {
      for (int x = startX; x < endX; x += _sampleStride) {
        final img.Pixel pixel = decoded.getPixel(x, y);

        final double r = pixel.r / 255.0;
        final double g = pixel.g / 255.0;
        final double b = pixel.b / 255.0;

        final String colorName = _rgbToColorName(r, g, b);
        votes[colorName] = (votes[colorName] ?? 0) + 1;
        rgbAccum[colorName] ??= [0, 0, 0, 0]; // r, g, b, count
        rgbAccum[colorName]![0] += pixel.r.toInt();
        rgbAccum[colorName]![1] += pixel.g.toInt();
        rgbAccum[colorName]![2] += pixel.b.toInt();
        rgbAccum[colorName]![3] += 1;
        totalSamples++;
      }
    }

    if (totalSamples == 0 || votes.isEmpty) {
      return const DetectedColor(
        name: 'Unknown',
        confidence: 0.0,
        red: 128,
        green: 128,
        blue: 128,
      );
    }

    // Pick the color with the most votes.
    final String winner = votes.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
    final double confidence = votes[winner]! / totalSamples;

    // Compute the average RGB for the winning color to use as the swatch.
    final List<int> acc = rgbAccum[winner]!;
    final int count = acc[3];
    final int swatchR = (acc[0] / count).round();
    final int swatchG = (acc[1] / count).round();
    final int swatchB = (acc[2] / count).round();

    return DetectedColor(
      name: winner,
      confidence: confidence,
      red: swatchR,
      green: swatchG,
      blue: swatchB,
    );
  }

  // ── HSV color naming ───────────────────────────────────────────────────────

  /// Converts a normalised RGB triplet (each in [0.0, 1.0]) to a color name.
  ///
  /// Steps:
  ///   1. Compute HSV.
  ///   2. Check achromatic cases first (black, white, gray) using Value and
  ///      Saturation thresholds.
  ///   3. Check for brown/beige (orange-ish hue, low saturation).
  ///   4. Map hue angle to one of 8 chromatic color names.
  static String _rgbToColorName(double r, double g, double b) {
    final double maxC = math.max(r, math.max(g, b));
    final double minC = math.min(r, math.min(g, b));
    final double delta = maxC - minC;

    // ── Value (brightness) based achromatic detection ─────────────────────
    if (maxC < 0.12) return 'Black';
    if (delta < 0.07) {
      if (maxC > 0.85) return 'White';
      if (maxC > 0.45) return 'Gray';
      return 'Dark Gray';
    }

    // ── Saturation check ──────────────────────────────────────────────────
    final double s = delta / maxC; // Saturation in [0, 1]
    if (s < 0.12) {
      if (maxC > 0.80) return 'White';
      if (maxC > 0.40) return 'Gray';
      return 'Dark Gray';
    }

    // ── Hue calculation ───────────────────────────────────────────────────
    double h;
    if (maxC == r) {
      h = ((g - b) / delta) % 6.0;
    } else if (maxC == g) {
      h = (b - r) / delta + 2.0;
    } else {
      h = (r - g) / delta + 4.0;
    }
    h = h * 60.0;
    if (h < 0) h += 360.0;

    // ── Brown / Beige (desaturated orange-yellow) ─────────────────────────
    if (h >= 15 && h < 50) {
      if (s < 0.45 && maxC < 0.72) return 'Brown';
      if (s < 0.30 && maxC >= 0.72) return 'Beige';
    }

    // ── Chromatic hue bands ───────────────────────────────────────────────
    if (h < 15 || h >= 345) return 'Red';
    if (h < 45) return 'Orange';
    if (h < 75) return 'Yellow';
    if (h < 155) return 'Green';
    if (h < 195) return 'Cyan';
    if (h < 255) return 'Blue';
    if (h < 295) return 'Purple';
    if (h < 345) return 'Pink';

    return 'Red'; // fallback for h ≥ 345
  }
}
