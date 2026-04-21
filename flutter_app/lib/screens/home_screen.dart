/// The first screen the user sees when the app launches.
///
/// Provides a short description of the app's purpose and an entry-point
/// button that navigates to [CameraScreen] to capture a clothing photo.

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'camera_screen.dart';

/// Home screen – application landing page.
class HomeScreen extends StatelessWidget {
  /// All cameras available on the device, passed in from [main].
  final List<CameraDescription> availableCamerasList;

  const HomeScreen({super.key, required this.availableCamerasList});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Clothing Recognizer'),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ── App icon / illustration ─────────────────────────────
              Icon(
                Icons.checkroom_outlined,
                size: 96,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 24),

              // ── Headline ────────────────────────────────────────────
              Text(
                'AI Clothing Detector',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),

              // ── Description ─────────────────────────────────────────
              Text(
                'Take a photo of a clothing item and the app will '
                'identify its type (T-shirt, Hoodie, Jeans, …) using '
                'a neural network trained on the Fashion-MNIST dataset.',
                style: Theme.of(context).textTheme.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 40),

              // ── Navigate to camera ───────────────────────────────────
              FilledButton.icon(
                onPressed: () => _openCameraScreen(context),
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Take a photo'),
                style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(52),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Pushes [CameraScreen] onto the navigation stack.
  void _openCameraScreen(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) =>
            CameraScreen(availableCamerasList: availableCamerasList),
      ),
    );
  }
}
