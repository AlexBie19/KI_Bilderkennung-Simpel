/// Reusable widget that displays the full [ClassificationResult].
///
/// Shows the best-match label prominently at the top followed by a ranked
/// list of up to three predictions, each with a labelled confidence bar.
/// Designed as a stateless widget so it can be embedded without side-effects.
library;

import 'package:flutter/material.dart';

import '../models/classification_result.dart';

/// A card-style widget that presents clothing classification results.
class ClassificationResultWidget extends StatelessWidget {
  final ClassificationResult classificationResult;

  const ClassificationResultWidget({
    super.key,
    required this.classificationResult,
  });

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    final TextTheme text = Theme.of(context).textTheme;

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Best match headline ──────────────────────────────────────
            Text(
              'Erkanntes Kleidungsstück',
              style: text.labelMedium?.copyWith(
                color: colors.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              classificationResult.clothingLabel,
              style: text.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: colors.primary,
              ),
            ),

            const SizedBox(height: 20),
            const Divider(),
            const SizedBox(height: 12),

            // ── Top-3 ranked predictions ─────────────────────────────────
            Text(
              'Top Ergebnisse',
              style: text.labelMedium?.copyWith(
                color: colors.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 10),

            ...classificationResult.topPredictions
                .asMap()
                .entries
                .map((entry) => _PredictionRow(
                      rank: entry.key + 1,
                      prediction: entry.value,
                      isTop: entry.key == 0,
                    )),
          ],
        ),
      ),
    );
  }
}

/// One row in the ranked prediction list.
class _PredictionRow extends StatelessWidget {
  const _PredictionRow({
    required this.rank,
    required this.prediction,
    required this.isTop,
  });

  final int rank;
  final ClothingPrediction prediction;
  final bool isTop;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;
    final TextTheme text = Theme.of(context).textTheme;

    final Color barColor =
        isTop ? colors.primary : colors.secondary.withValues(alpha: 0.7);

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              // Rank badge
              Container(
                width: 24,
                height: 24,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: isTop ? colors.primary : colors.surfaceContainerHighest,
                  shape: BoxShape.circle,
                ),
                child: Text(
                  '$rank',
                  style: text.labelSmall?.copyWith(
                    color: isTop
                        ? colors.onPrimary
                        : colors.onSurfaceVariant,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // Label
              Expanded(
                child: Text(
                  prediction.label,
                  style: text.bodyMedium?.copyWith(
                    fontWeight:
                        isTop ? FontWeight.w600 : FontWeight.normal,
                  ),
                ),
              ),
              // Percentage
              Text(
                prediction.formattedConfidence,
                style: text.bodySmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: colors.onSurfaceVariant,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: prediction.confidence,
              minHeight: 8,
              backgroundColor: colors.surfaceContainerHighest,
              valueColor: AlwaysStoppedAnimation<Color>(barColor),
            ),
          ),
        ],
      ),
    );
  }
}
