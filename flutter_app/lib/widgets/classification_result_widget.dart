/// Reusable widget that displays the full [ClassificationResult].
///
/// Shows the best-match clothing label prominently at the top, followed by a
/// color swatch + name row (when color information is available), and then a
/// ranked list of up to three predictions, each with a labelled confidence bar.
/// Designed as a stateless widget so it can be embedded without side-effects.
library;

import 'package:flutter/material.dart';

import '../models/classification_result.dart';
import '../services/color_analysis_service.dart';

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

            // ── Detected color (shown only when available) ───────────────
            if (classificationResult.detectedColor != null) ...[
              const SizedBox(height: 16),
              _ColorRow(detectedColor: classificationResult.detectedColor!),
            ],

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

/// Row that shows a color swatch circle and the detected color name.
class _ColorRow extends StatelessWidget {
  const _ColorRow({required this.detectedColor});

  final DetectedColor detectedColor;

  @override
  Widget build(BuildContext context) {
    final TextTheme text = Theme.of(context).textTheme;
    final ColorScheme colors = Theme.of(context).colorScheme;

    final Color swatchColor = Color.fromRGBO(
      detectedColor.red,
      detectedColor.green,
      detectedColor.blue,
      1.0,
    );

    return Row(
      children: [
        // Color swatch circle with a subtle border so white/light colors
        // are visible against the card background.
        Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: swatchColor,
            shape: BoxShape.circle,
            border: Border.all(
              color: colors.outlineVariant,
              width: 1.5,
            ),
          ),
        ),
        const SizedBox(width: 12),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Erkannte Farbe',
              style: text.labelSmall?.copyWith(
                color: colors.onSurfaceVariant,
              ),
            ),
            Text(
              detectedColor.name,
              style: text.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        const Spacer(),
        Text(
          detectedColor.formattedConfidence,
          style: text.bodySmall?.copyWith(
            color: colors.onSurfaceVariant,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
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
