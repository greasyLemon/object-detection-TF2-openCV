import cv2
import numpy as np
from tensorflow.lite.runtime import Interpreter

# Load TFLite model
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input details (assuming model has a single input tensor)
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

# Open webcam
cap = cv2.VideoCapture(0)

while True:
  # Read frame from webcam
  ret, frame = cap.read()
  if not ret:
    break

  # Preprocess frame (resize, normalize, etc.) based on your model's input requirements
  preprocessed_frame = preprocess_frame(frame, input_shape)

  # Set input tensor data
  interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

  # Run inference
  interpreter.invoke()

  # Get output details (assuming model has a single output tensor)
  output_details = interpreter.get_output_details()
  output_data = interpreter.get_tensor(output_details[0]['index'])

  # Process output data (reshape, post-processing, etc.) based on your model's output format
  results = process_output(output_data)

  # Draw or display results on the frame (optional)
  draw_results(frame, results)

  # Display webcam frame
  cv2.imshow('Webcam Feed with TFLite Inference', frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()