/// Service that loads the MobileNetV2-based TFLite clothing classifier and
/// runs inference on a preprocessed image tensor.
///
/// Architecture decision
/// ---------------------
/// The class exposes a single async factory constructor [ClothingClassifierService.load]
/// that initialises the interpreter and reads the label list from the asset
/// bundle.  Callers must [dispose] the service when it is no longer needed
/// to free native TFLite resources.
///
/// Input / output format
/// ---------------------
/// • Input  : flat [Float32List] of size 128 × 128 × 3 = 49 152 values in
///            row-major, channel-last order, values in [−1.0, 1.0].
///            Produced by [ImagePreprocessingService.preprocessImageForMobileNetV2].
/// • Output : 10 softmax probabilities (one per Fashion-MNIST class).
///
/// Extensibility
/// -------------
/// To add color or pattern recognition, create a parallel service that loads
/// a different TFLite model file and returns an extended result type.  Both
/// services can be composed in the screen layer without changing this file.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/classification_result.dart';

/// Loads the MobileNetV2 clothing classifier model and performs inference.
class ClothingClassifierService {
  // ── Asset paths ───────────────────────────────────────────────────────────

  static const String _modelAssetPath =
      'assets/models/fashion_mnist_model.tflite';

  static const String _labelsAssetPath = 'assets/models/labels.txt';

  // ── Internal state ────────────────────────────────────────────────────────

  final Interpreter _tfliteInterpreter;
  final List<String> _clothingLabels;

  // ─────────────────────────────────────────────────────────────────────────

  ClothingClassifierService._({
    required Interpreter tfliteInterpreter,
    required List<String> clothingLabels,
  })  : _tfliteInterpreter = tfliteInterpreter,
        _clothingLabels = clothingLabels;

  /// Asynchronously loads the TFLite model and label file and returns a
  /// ready-to-use [ClothingClassifierService].
  ///
  /// Throws an [Exception] if either asset cannot be loaded.
  static Future<ClothingClassifierService> load() async {
    final Interpreter interpreter = await Interpreter.fromAsset(
      _modelAssetPath,
      options: InterpreterOptions()..threads = 1,
    );
    // Explicitly set input shape [1, 128, 128, 3] before allocating.
    // Without this call, TFLite may not finalize tensor shapes and throws
    // 'Failed precondition' when invoke() is called.
    interpreter.resizeInputTensor(0, [1, 128, 128, 3]);
    interpreter.allocateTensors();

    final String labelsContent = await rootBundle.loadString(_labelsAssetPath);
    final List<String> labels = labelsContent
        .split('\n')
        .map((String l) => l.trim())
        .where((String l) => l.isNotEmpty && !l.startsWith('#'))
        .toList();

    return ClothingClassifierService._(
      tfliteInterpreter: interpreter,
      clothingLabels: labels,
    );
  }

  /// Runs inference on [inputTensor] and returns a [ClassificationResult]
  /// containing the top-3 predictions sorted by confidence (highest first).
  ///
  /// [inputTensor] must be the [Float32List] produced by
  /// [ImagePreprocessingService.preprocessImageForMobileNetV2]
  /// (49 152 values, shape 128 x 128 x 3, range [-1, 1]).
  ClassificationResult classifyClothing(Float32List inputTensor) {
    // Determine the model's actual output size from its tensor metadata so
    // the code stays correct even if the label count changes.
    final int numberOfClasses =
        _tfliteInterpreter.getOutputTensor(0).shape.last;

    // Copy input bytes directly into the model's input tensor and invoke.
    // Using invoke() instead of run() avoids type-mismatch issues that
    // arise when passing a flat Float32List to run() in tflite_flutter 0.12.x.
    try {
      final Uint8List inputBytes = inputTensor.buffer.asUint8List();
      _tfliteInterpreter
          .getInputTensor(0)
          .data
          .setRange(0, inputBytes.length, inputBytes);
      _tfliteInterpreter.invoke();
    } catch (e) {
      throw StateError(
        'TFLite inference failed: $e. '
        'Input tensor length: ${inputTensor.length} '
        '(expected ${128 * 128 * 3} for 128x128x3). '
        'Output shape: ${_tfliteInterpreter.getOutputTensor(0).shape}',
      );
    }

    // Read the output directly from the output tensor buffer.
    final Float32List rawOutput =
        _tfliteInterpreter.getOutputTensor(0).data.buffer.asFloat32List();
    final List<double> probabilities = rawOutput.toList();

    // Build top-3 predictions (sorted by confidence, descending).
    final List<_IndexedScore> indexed = List.generate(
      probabilities.length,
      (i) => _IndexedScore(i, probabilities[i]),
    )..sort((a, b) => b.score.compareTo(a.score));

    final List<ClothingPrediction> topPredictions = indexed
        .take(3)
        .map(
          (s) => ClothingPrediction(
            classIndex: s.index,
            label: s.index < _clothingLabels.length
                ? _clothingLabels[s.index]
                : 'Unknown',
            confidence: s.score,
          ),
        )
        .toList();

    return ClassificationResult(topPredictions: topPredictions);
  }

  /// Releases native TFLite resources.  Must be called when the service is
  /// no longer needed (e.g. in the widget's [dispose] method).
  void dispose() {
    _tfliteInterpreter.close();
  }
}

/// Internal helper that pairs a class index with its softmax score.
class _IndexedScore {
  const _IndexedScore(this.index, this.score);
  final int index;
  final double score;
}
