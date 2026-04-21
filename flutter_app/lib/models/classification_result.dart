/// Data model that holds the result of a single clothing classification.
///
/// Designed to be extensible: future versions can add [detectedColor] and
/// [detectedPattern] fields without breaking existing code.

// ignore_for_file: public_member_api_docs

/// Represents one classification result returned by [ClothingClassifierService].
class ClassificationResult {
  /// The human-readable clothing label (e.g. "T-shirt/top").
  final String clothingLabel;

  /// Confidence score in the range [0.0, 1.0].
  final double confidenceScore;

  /// Index into the label list that corresponds to [clothingLabel].
  final int classIndex;

  // ── Future extension fields (currently unused) ─────────────────────────────
  // final String? detectedColor;   // e.g. "Red", "Blue"
  // final String? detectedPattern; // e.g. "Striped", "Checkered"
  // ───────────────────────────────────────────────────────────────────────────

  /// Creates a [ClassificationResult].
  ///
  /// All parameters are required; there are no nullable fields at this stage
  /// so callers always receive a complete result object.
  const ClassificationResult({
    required this.clothingLabel,
    required this.confidenceScore,
    required this.classIndex,
  });

  /// Returns a formatted percentage string for display purposes.
  ///
  /// Example: `confidenceScore = 0.942` → `"94.2 %"`
  String get formattedConfidence =>
      '${(confidenceScore * 100).toStringAsFixed(1)} %';

  @override
  String toString() =>
      'ClassificationResult('
      'clothingLabel: $clothingLabel, '
      'confidenceScore: $confidenceScore, '
      'classIndex: $classIndex'
      ')';
}
