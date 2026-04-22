/// Entry point of the Clothing Recognizer Flutter application.
///
/// Initialises the camera list before the widget tree is built, then
/// launches [ClothingRecognizerApp].
library;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

/// List of all available cameras discovered at app startup.
/// Passed down to the camera screen so it does not have to re-query.
late List<CameraDescription> availableCamerasList;

Future<void> main() async {
  // Ensure Flutter bindings are initialised before accessing plugins.
  WidgetsFlutterBinding.ensureInitialized();

  // Retrieve all cameras available on the device (front, back, external).
  availableCamerasList = await availableCameras();

  runApp(const ClothingRecognizerApp());
}

/// Root widget of the application.
///
/// Sets up the [MaterialApp] with a neutral theme and points to
/// [HomeScreen] as the initial route.
class ClothingRecognizerApp extends StatelessWidget {
  const ClothingRecognizerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Clothing Recognizer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        // Use a blue seed colour; easily changed in future iterations.
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      home: HomeScreen(availableCamerasList: availableCamerasList),
    );
  }
}
