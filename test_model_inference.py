#!/usr/bin/env python3
"""
Test script to verify the exported TFLite model works correctly.
This mimics exactly what the Flutter app does:
  1. Create a test image (or load from file)
  2. Preprocess: resize to 128×128, normalize to [-1, 1]
  3. Run inference
  4. Print top predictions
"""

import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

def preprocess_image(image_input, target_size=128):
    """Preprocess image exactly like ImagePreprocessingService.dart does."""
    # Handle both file paths and PIL Image objects
    if isinstance(image_input, str):
        # Load from file
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        # Already a PIL Image
        img = image_input.convert('RGB')
    else:
        # Assume it's a numpy array
        img = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
    
    # Center-crop to square
    width, height = img.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    
    # Resize to target_size × target_size
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize to [-1, 1]
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 127.5 - 1.0
    
    return img_array

def create_test_image_red_shirt():
    """Create a simple test image: red rectangle on white background."""
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    # Draw a red rectangle (simulating a red t-shirt)
    draw.rectangle([50, 50, 200, 200], fill='red')
    return img

def run_inference(model_path, image_array):
    """Run TFLite inference on a preprocessed image."""
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"Model input shape: {input_details['shape']}")
    print(f"Model input dtype: {input_details['dtype']}")
    print(f"Model output shape: {output_details['shape']}")
    
    # Prepare input: add batch dimension
    if len(image_array.shape) == 3:
        input_tensor = np.expand_dims(image_array, axis=0).astype(np.float32)
    else:
        input_tensor = image_array.astype(np.float32)
    
    print(f"\nInput tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Set input tensor
    interpreter.set_tensor(input_details['index'], input_tensor)
    
    # Run inference
    try:
        interpreter.invoke()
        print("[OK] Inference succeeded!")
    except Exception as e:
        print(f"[ERROR] Inference FAILED: {e}")
        return None
    
    # Get output
    output_tensor = interpreter.get_tensor(output_details['index'])
    print(f"\nOutput tensor shape: {output_tensor.shape}")
    print(f"Output probabilities sum: {output_tensor[0].sum():.6f} (should be ~1.0)")
    
    return output_tensor[0]

def main():
    model_path = 'flutter_app/assets/models/fashion_mnist_model.tflite'
    labels_path = 'flutter_app/assets/models/labels.txt'
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Loaded {len(labels)} labels: {labels}\n")
    
    # Test 1: Create a synthetic red t-shirt image
    print("=" * 60)
    print("TEST 1: Synthetic red rectangle (simulating red t-shirt)")
    print("=" * 60)
    test_img = create_test_image_red_shirt()
    test_img.save('test_red_tshirt.png')
    print(f"[OK] Created test image: test_red_tshirt.png\n")
    
    # Preprocess
    preprocessed = preprocess_image(test_img)
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]\n")
    
    # Run inference
    probabilities = run_inference(model_path, preprocessed)
    
    if probabilities is not None:
        print("\nTop 3 predictions:")
        top_indices = np.argsort(probabilities)[::-1][:3]
        for rank, idx in enumerate(top_indices, 1):
            prob = probabilities[idx]
            label = labels[idx] if idx < len(labels) else "Unknown"
            print(f"  {rank}. {label}: {prob:.2%}")
    
    print()

if __name__ == '__main__':
    main()
