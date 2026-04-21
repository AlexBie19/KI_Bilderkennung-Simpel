/// Reusable widget that displays a single [ClassificationResult].
///
/// Shows the detected clothing label, a confidence percentage bar, and
/// the raw confidence value.  Designed as a stateless widget so it can
/// be embedded in any screen without side-effects.

import 'package:flutter/material.dart';

import '../models/classification_result.dart';

/// A card-style widget that presents one clothing classification result.
class ClassificationResultWidget extends StatelessWidget {
  /// The classification result to display.
  final ClassificationResult classificationResult;

  const ClassificationResultWidget({
    super.key,
    required this.classificationResult,
  });

  @override
  Widget build(BuildContext context) {
    final ColorScheme colorScheme = Theme.of(context).colorScheme;
    final TextTheme textTheme = Theme.of(context).textTheme;

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Clothing label ──────────────────────────────────────────
            Text(
              'Detected clothing type',
              style: textTheme.labelMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              classificationResult.clothingLabel,
              style: textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: colorScheme.primary,
              ),
            ),

            const SizedBox(height: 16),

            // ── Confidence bar ──────────────────────────────────────────
            Text(
              'Confidence',
              style: textTheme.labelMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                // confidenceScore is in [0.0, 1.0] – perfect for the value
                value: classificationResult.confidenceScore,
                minHeight: 12,
                backgroundColor: colorScheme.surfaceContainerHighest,
                valueColor:
                    AlwaysStoppedAnimation<Color>(colorScheme.primary),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              classificationResult.formattedConfidence,
              style: textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),

            const SizedBox(height: 12),

            // ── Class index (useful for debugging) ──────────────────────
            Text(
              'Class index: ${classificationResult.classIndex}',
              style: textTheme.bodySmall?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
