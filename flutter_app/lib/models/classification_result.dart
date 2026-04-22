/// Data models for clothing classification results.
///
/// [ClassificationResult] holds the full list of top-k predictions returned
/// by [ClothingClassifierService].  The first element in [topPredictions] is
/// always the highest-confidence class.
///
/// Designed to be extensible: future versions can add [detectedColor] and
/// [detectedPattern] fields to [ClassificationResult] without breaking any
/// existing code that reads [topPrediction].
library;

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
/// sorted by confidence (highest first).
class ClassificationResult {
  /// Ordered list of predictions (index 0 = best match).
  final List<ClothingPrediction> topPredictions;

  const ClassificationResult({required this.topPredictions});

  /// The highest-confidence prediction.  Throws if [topPredictions] is empty.
  ClothingPrediction get topPrediction => topPredictions.first;

  /// Convenience alias for the best label.
  String get clothingLabel => topPrediction.label;

  /// Convenience alias for the best confidence score.
  double get confidenceScore => topPrediction.confidence;

  /// Convenience alias for the best class index.
  int get classIndex => topPrediction.classIndex;

  @override
  String toString() => 'ClassificationResult(top: $topPrediction)';
}
