/// Service that loads the Fashion-MNIST TFLite model and runs inference
/// on a preprocessed image tensor to identify the type of clothing.
///
/// Architecture decision:
///   The class exposes a single async factory constructor [ClothingClassifierService.load]
///   that initialises the interpreter and reads the label list from the asset
///   bundle.  Callers must [dispose] the service when it is no longer needed
///   to free native TFLite resources.
///
/// Extension note: to add color or pattern recognition, create a parallel
/// service (e.g. `ColorPatternClassifierService`) that loads a different
/// TFLite model and returns an extended result type.  Both services can
/// be composed in the screen layer without modifying this file.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/classification_result.dart';

/// Loads the Fashion-MNIST clothing classifier model and performs inference.
class ClothingClassifierService {
  // ── Model asset paths (declared as constants for easy maintenance) ────────

  /// Asset path for the TFLite model file.
  static const String _modelAssetPath =
      'assets/models/fashion_mnist_model.tflite';

  /// Asset path for the plain-text label file (one label per line).
  static const String _labelsAssetPath = 'assets/models/labels.txt';

  // ── Internal state ────────────────────────────────────────────────────────

  /// Native TFLite interpreter instance.
  final Interpreter _tfliteInterpreter;

  /// Ordered list of class names loaded from [_labelsAssetPath].
  /// Index 0 corresponds to output neuron 0, etc.
  final List<String> _clothingLabels;

  // ─────────────────────────────────────────────────────────────────────────

  /// Private constructor – callers must use [ClothingClassifierService.load].
  ClothingClassifierService._({
    required Interpreter tfliteInterpreter,
    required List<String> clothingLabels,
  })  : _tfliteInterpreter = tfliteInterpreter,
        _clothingLabels = clothingLabels;

  /// Asynchronously loads the TFLite model and label file from the Flutter
  /// asset bundle and returns a ready-to-use [ClothingClassifierService].
  ///
  /// Throws an [Exception] if either asset cannot be loaded.
  static Future<ClothingClassifierService> load() async {
    // Load the TFLite model from assets.
    final Interpreter loadedInterpreter =
        await Interpreter.fromAsset(_modelAssetPath);

    // Load and parse the labels file; split on newlines, remove blank lines.
    final String labelsFileContent =
        await rootBundle.loadString(_labelsAssetPath);
    final List<String> parsedLabels = labelsFileContent
        .split('\n')
        .map((String line) => line.trim())
        .where((String line) => line.isNotEmpty)
        .toList();

    return ClothingClassifierService._(
      tfliteInterpreter: loadedInterpreter,
      clothingLabels: parsedLabels,
    );
  }

  /// Runs the Fashion-MNIST model on [preprocessedInputTensor] and returns
  /// a [ClassificationResult] with the most likely clothing class.
  ///
  /// [preprocessedInputTensor] must be a [Float32List] of exactly 784 values
  /// (28 × 28 normalised grayscale pixels) as produced by
  /// [ImagePreprocessingService.preprocessImageForFashionMnist].
  ///
  /// The method picks the class with the highest softmax output score
  /// (argmax over the output tensor).
  ClassificationResult classifyClothing(Float32List preprocessedInputTensor) {
    // --- Prepare input tensor: shape [1, 28, 28, 1] ----------------------
    // TFLite expects a 4-D tensor: [batch, height, width, channels].
    // We have a single image (batch=1) of 28×28 grayscale pixels (channels=1).
    final List<List<List<List<double>>>> modelInput = List.generate(
      1, // batch size
      (_) => List.generate(
        28, // height
        (rowIndex) => List.generate(
          28, // width
          (colIndex) => [
            // Single grayscale channel value
            preprocessedInputTensor[rowIndex * 28 + colIndex].toDouble(),
          ],
        ),
      ),
    );

    // --- Prepare output tensor: shape [1, numberOfClasses] ---------------
    final int numberOfClasses = _clothingLabels.length;
    final List<List<double>> modelOutput = List.generate(
      1, // batch size
      (_) => List.filled(numberOfClasses, 0.0),
    );

    // --- Run inference ---------------------------------------------------
    _tfliteInterpreter.run(modelInput, modelOutput);

    // --- Find the class with the highest score (argmax) ------------------
    final List<double> classProbabilities = modelOutput[0];
    int bestClassIndex = 0;
    double highestScore = classProbabilities[0];

    for (int index = 1; index < classProbabilities.length; index++) {
      if (classProbabilities[index] > highestScore) {
        highestScore = classProbabilities[index];
        bestClassIndex = index;
      }
    }

    return ClassificationResult(
      classIndex: bestClassIndex,
      clothingLabel: _clothingLabels[bestClassIndex],
      confidenceScore: highestScore,
    );
  }

  /// Releases the native TFLite interpreter resources.
  /// Must be called when the service is no longer needed (e.g. widget dispose).
  void dispose() {
    _tfliteInterpreter.close();
  }
}
