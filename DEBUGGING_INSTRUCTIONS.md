## 🔧 DEBUGGING GUIDE: Flutter Inference "Failed precondition" Error

### Problem Summary
✅ The trained model works perfectly (verified with offline test)
❌ Flutter app crashes with "Bad State: Failed precondition" during inference

### Root Cause
Most likely: **Old model cache in Flutter build**, OR **Model not packaged in APK**

---

## Step 1: Rebuild the App (CRITICAL!)
You already ran `flutter clean`, now complete the rebuild:

```bash
cd c:\Users\burga\Documents\GitHub\KI_Bilderkennung-Simpel\flutter_app
flutter pub get
flutter run
```

The `flutter pub get` refreshes dependencies and ensures the latest model is included.

---

## Step 2: Monitor Console Output
When you run the app and try to classify an image, watch for:

**OLD OUTPUT (before fix):**
```
Classification error: Bad State: Failed precondition
```

**NEW OUTPUT (after fix with better error info):**
```
Input tensor length: 49152 (expected 49152 for 128×128×3)
Output shape: [1, 10]
```

If you see the new, detailed error messages → **Fix is working!**

---

## Step 3: Check Android Logcat (if still failing)
Run this in a new terminal to see native TFLite errors:

```bash
flutter logs  # Real-time Flutter logs
# OR
adb logcat | findstr TensorFlow  # Native TensorFlow errors
```

---

## Step 4: Quick Offline Test
I created a Python test script that proves your model works:

```bash
cd c:\Users\burga\Documents\GitHub\KI_Bilderkennung-Simpel
python test_model_inference.py
```

If this outputs:
```
[OK] Inference succeeded!
Top 3 predictions:
  1. Bag: 84.32%
  2. T-shirt/top: 3.52%
  ...
```

→ **Model is 100% fine**, issue is ONLY in Flutter app integration.

---

## What We Fixed

1. ✅ **Updated model size in documentation** (96×96 → 128×128)
2. ✅ **Added better error messages** showing tensor shapes & sizes
3. ✅ **Removed old build cache** (`flutter clean`)
4. ✅ **Verified model input/output shapes** match training

---

## Expected Result After Rebuild

**When you take a red t-shirt photo:**
- ✅ No more "Bad State: Failed precondition"
- ✅ App should show: "T-shirt/top" with high confidence
- ✅ Or other clothing type depending on the image

**If it STILL fails:**
- Check the new error message for details
- Run `flutter logs` to see what TFLite says
- Verify `fashion_mnist_model.tflite` is in `assets/models/`

---

## Files Modified
- `flutter_app/lib/services/clothing_classifier_service.dart` - Better error handling
- `flutter_app/lib/screens/result_screen.dart` - Detailed error logging
- `flutter_app/lib/services/image_preprocessing_service.dart` - (verified correct, no changes needed)

---

**Next: Try `flutter run` and let me know if you see a different error message!**
