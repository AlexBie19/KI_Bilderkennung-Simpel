/// Camera screen – displays a live preview from the device camera and lets
/// the user capture a photo that will be classified by the AI model.
///
/// Permission handling:
///   Before the camera is initialised the screen explicitly requests the
///   CAMERA permission via [permission_handler].  If the user denies the
///   permission, a human-readable error message is shown instead of crashing.
///
/// Lifecycle:
///   [CameraController] is created in [initState] and disposed in [dispose].
library;

import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'result_screen.dart';

/// Stateful screen that manages the camera lifecycle and photo capture.
class CameraScreen extends StatefulWidget {
  /// All cameras available on the device (from [main.availableCamerasList]).
  final List<CameraDescription> availableCamerasList;

  const CameraScreen({super.key, required this.availableCamerasList});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  // ── Camera controller ─────────────────────────────────────────────────────

  /// Controls the selected physical camera (initialised in [_initialiseCamera]).
  CameraController? _cameraController;

  /// Future that resolves once [_cameraController] is ready to preview.
  Future<void>? _cameraInitialisationFuture;

  // ── UI state ──────────────────────────────────────────────────────────────

  /// Human-readable error shown when camera or permission setup fails.
  String? _errorMessage;

  /// True while a photo capture is in progress (disables the capture button).
  bool _isCapturingPhoto = false;

  // ─────────────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    // Observe app lifecycle changes so the camera is paused/resumed correctly.
    WidgetsBinding.instance.addObserver(this);
    _requestCameraPermissionAndInitialise();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Release the camera hardware so other apps can use it.
    _cameraController?.dispose();
    super.dispose();
  }

  // ── App lifecycle ─────────────────────────────────────────────────────────

  @override
  void didChangeAppLifecycleState(AppLifecycleState lifecycleState) {
    final CameraController? controller = _cameraController;

    // Do nothing if the controller has not been created yet.
    if (controller == null || !controller.value.isInitialized) return;

    if (lifecycleState == AppLifecycleState.inactive) {
      // Release camera when app goes to background.
      controller.dispose();
    } else if (lifecycleState == AppLifecycleState.resumed) {
      // Re-initialise camera when app comes back to foreground.
      _initialiseCamera(controller.description);
    }
  }

  // ── Permission & camera setup ─────────────────────────────────────────────

  /// Requests camera permission; on success calls [_initialiseCamera].
  Future<void> _requestCameraPermissionAndInitialise() async {
    final PermissionStatus cameraPermissionStatus =
        await Permission.camera.request();

    if (!mounted) return;

    if (cameraPermissionStatus.isGranted) {
      if (widget.availableCamerasList.isEmpty) {
        setState(() => _errorMessage = 'No camera found on this device.');
        return;
      }
      // Use the first available camera (usually the rear-facing camera).
      _initialiseCamera(widget.availableCamerasList.first);
    } else {
      setState(() {
        _errorMessage = 'Camera permission denied.\n'
            'Please enable it in your device settings.';
      });
    }
  }

  /// Creates and initialises a [CameraController] for [selectedCamera].
  void _initialiseCamera(CameraDescription selectedCamera) {
    final CameraController newCameraController = CameraController(
      selectedCamera,
      // ResolutionPreset.high gives a sharp live preview.
      // Downscaling to 128x128 for MobileNetV2 happens in ImagePreprocessingService
      // *after* capture, so the user always sees a crisp full-resolution view.
      ResolutionPreset.high,
      enableAudio: false, // Audio is not needed for image classification.
    );

    _cameraController = newCameraController;
    _cameraInitialisationFuture = newCameraController.initialize().then((_) {
      // Trigger a rebuild so the preview becomes visible.
      if (mounted) setState(() {});
    }).catchError((Object cameraError) {
      if (mounted) {
        setState(() => _errorMessage = 'Camera initialisation failed: '
            '$cameraError');
      }
    });
  }

  // ── Photo capture ─────────────────────────────────────────────────────────

  /// Captures a photo and navigates to [ResultScreen] with the image file.
  Future<void> _capturePhoto() async {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    setState(() => _isCapturingPhoto = true);

    try {
      // takePicture() saves the image to a temporary file on the device.
      final XFile capturedImageFile = await controller.takePicture();

      if (!mounted) return;

      // Navigate to the result screen, passing the captured image path.
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) =>
              ResultScreen(capturedImageFile: File(capturedImageFile.path)),
        ),
      );
    } catch (captureError) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content:
                  Text('Foto konnte nicht aufgenommen werden: $captureError')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturingPhoto = false);
    }
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Foto aufnehmen'),
      ),
      body: _buildCameraBody(),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: _buildCaptureButton(),
    );
  }

  /// Builds the main camera body: handles loading, error, or live preview.
  Widget _buildCameraBody() {
    // Show error message when permissions are denied or camera is unavailable.
    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Text(
            _errorMessage!,
            style: const TextStyle(color: Colors.white, fontSize: 16),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    // Show a loading spinner while the camera controller initialises.
    if (_cameraInitialisationFuture == null) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }

    return FutureBuilder<void>(
      future: _cameraInitialisationFuture,
      builder: (BuildContext context, AsyncSnapshot<void> snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          // Camera is ready – show the live preview, filling the screen.
          return CameraPreview(_cameraController!);
        } else {
          return const Center(
            child: CircularProgressIndicator(color: Colors.white),
          );
        }
      },
    );
  }

  /// Builds the circular capture button at the bottom of the screen.
  Widget _buildCaptureButton() {
    return FloatingActionButton.large(
      onPressed: _isCapturingPhoto ? null : _capturePhoto,
      backgroundColor: Colors.white,
      foregroundColor: Colors.black,
      tooltip: 'Foto aufnehmen',
      child: _isCapturingPhoto
          ? const CircularProgressIndicator(color: Colors.black)
          : const Icon(Icons.camera, size: 40),
    );
  }
}
