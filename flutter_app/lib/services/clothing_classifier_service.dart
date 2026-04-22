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
/// • Input  : flat [Float32List] of size 96 × 96 × 3 = 27 648 values in
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
      options: InterpreterOptions()..threads = 2,
    );
    // Allocate input / output tensors based on the model's shape metadata.
    interpreter.allocateTensors();

    final String labelsContent =
        await rootBundle.loadString(_labelsAssetPath);
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
  /// (27 648 values, shape 96 × 96 × 3, range [−1, 1]).
  ClassificationResult classifyClothing(Float32List inputTensor) {
    // Determine the model's actual output size from its tensor metadata so
    // the code stays correct even if the label count changes.
    final int numberOfClasses = _tfliteInterpreter
        .getOutputTensor(0)
        .shape
        .last;

    // Prepare output buffer: shape [1, numberOfClasses].
    final List<List<double>> outputBuffer = List.generate(
      1,
      (_) => List.filled(numberOfClasses, 0.0),
    );

    // Pass the flat Float32List directly – tflite_flutter maps it to the
    // model's [1, 96, 96, 3] input tensor automatically.
    _tfliteInterpreter.run(inputTensor, outputBuffer);

    final List<double> probabilities = outputBuffer[0];

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
