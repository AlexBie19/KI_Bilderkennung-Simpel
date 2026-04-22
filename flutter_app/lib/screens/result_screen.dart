/// Result screen – loads the clothing classifier, preprocesses the captured
/// image, runs inference, and displays the [ClassificationResultWidget].
///
/// The screen handles its own async loading state so the user always sees
/// either a spinner, an error message, or the final result.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/classification_result.dart';
import '../services/clothing_classifier_service.dart';
import '../services/image_preprocessing_service.dart';
import '../widgets/classification_result_widget.dart';

/// Displays the AI classification result for a single captured [File].
class ResultScreen extends StatefulWidget {
  /// The image file captured by [CameraScreen].
  final File capturedImageFile;

  const ResultScreen({super.key, required this.capturedImageFile});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  // ── Async state ───────────────────────────────────────────────────────────

  /// True while the model is loading or inference is running.
  bool _isLoading = true;

  /// Populated when classification succeeds.
  ClassificationResult? _classificationResult;

  /// Populated when loading or inference throws an exception.
  String? _errorMessage;

  // ─────────────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _runClassification();
  }

  /// Loads the classifier, preprocesses the image, and runs inference.
  Future<void> _runClassification() async {
    ClothingClassifierService? classifierService;

    try {
      // --- Step 1: Load the TFLite model and label file from assets ------
      classifierService = await ClothingClassifierService.load();

      // --- Step 2: Preprocess the captured image -------------------------
      // Resizes to 96×96 RGB, normalises to [−1, 1] (MobileNetV2 format).
      final Float32List preprocessedInputTensor =
          ImagePreprocessingService.preprocessImageForMobileNetV2(
        widget.capturedImageFile,
      );

      // --- Step 3: Run inference -----------------------------------------
      final ClassificationResult result =
          classifierService.classifyClothing(preprocessedInputTensor);

      if (mounted) {
        setState(() {
          _classificationResult = result;
          _isLoading = false;
        });
      }
    } catch (classificationError) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Klassifizierung fehlgeschlagen:\n$classificationError';
          _isLoading = false;
        });
      }
    } finally {
      // Always release the interpreter even if an error occurred.
      classifierService?.dispose();
    }
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ergebnis'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Captured image preview ────────────────────────────────
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.file(
                widget.capturedImageFile,
                height: 280,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(height: 24),

            // ── Result area ───────────────────────────────────────────
            _buildResultArea(),

            const SizedBox(height: 24),

            // ── Take another photo button ─────────────────────────────
            OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.camera_alt_outlined),
              label: const Text('Weiteres Foto aufnehmen'),
            ),
          ],
        ),
      ),
    );
  }

  /// Builds the result area based on the current async state.
  Widget _buildResultArea() {
    if (_isLoading) {
      // Show spinner and a descriptive message while inference is running.
      return const Column(
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 16),
          Text(
            'Kleidungsstück wird analysiert…',
            textAlign: TextAlign.center,
          ),
        ],
      );
    }

    if (_errorMessage != null) {
      return Text(
        _errorMessage!,
        style: TextStyle(color: Theme.of(context).colorScheme.error),
        textAlign: TextAlign.center,
      );
    }

    // Both _isLoading == false and _classificationResult != null at this point.
    return ClassificationResultWidget(
      classificationResult: _classificationResult!,
    );
  }
}
