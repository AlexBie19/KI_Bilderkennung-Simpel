/// Data models for clothing classification results.
///
/// [ClassificationResult] holds the full list of top-k predictions returned
/// by [ClothingClassifierService].  The first element in [topPredictions] is
/// always the highest-confidence class.
///
/// Extensible design: [ClassificationResult] now carries an optional
/// [detectedColor] field populated by [ColorAnalysisService] so the result
/// screen can show both the clothing type *and* the dominant color from the
/// same photo without changing the classifier model.
library;

import 'services/color_analysis_service.dart';

/// A single class prediction: label, confidence score and class index.
class ClothingPrediction {
  /// Human-readable label (e.g. "T-shirt/top").
  final String label;

  /// Softmax probability in [0.0, 1.0].
  final double confidence;

  /// Index into the Fashion-MNIST label list (0–9).
  final int classIndex;

  const ClothingPrediction({
    required this.label,
    required this.confidence,
    required this.classIndex,
  });

  /// Returns a formatted percentage string, e.g. "94.2 %".
  String get formattedConfidence =>
      '${(confidence * 100).toStringAsFixed(1)} %';

  @override
  String toString() =>
      'ClothingPrediction(label: $label, confidence: $confidence, '
      'classIndex: $classIndex)';
}

/// The result of one classification run, containing the top-k predictions
/// sorted by confidence (highest first) and an optional detected color.
class ClassificationResult {
  /// Ordered list of predictions (index 0 = best match).
  final List<ClothingPrediction> topPredictions;

  /// Dominant color detected in the photo by [ColorAnalysisService].
  /// Null if color analysis failed or was not performed.
  final DetectedColor? detectedColor;

  const ClassificationResult({
    required this.topPredictions,
    this.detectedColor,
  });

  /// The highest-confidence prediction.  Throws if [topPredictions] is empty.
  ClothingPrediction get topPrediction => topPredictions.first;

  /// Convenience alias for the best label.
  String get clothingLabel => topPrediction.label;

  /// Convenience alias for the best confidence score.
  double get confidenceScore => topPrediction.confidence;

  /// Convenience alias for the best class index.
  int get classIndex => topPrediction.classIndex;

  @override
  String toString() =>
      'ClassificationResult(top: $topPrediction, color: $detectedColor)';
}
