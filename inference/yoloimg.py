import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# YOLO object detection setup
model = YOLO("D:/Games/best.pt")  # Path to your YOLO model

# Color detection settings (adjust these based on your needs)
green_lower = np.array([25, 50, 50])  # Lower HSV bounds for green
green_upper = np.array([95, 255, 255])  # Upper HSV bounds for green

def preprocess_image(img, target_size=(640, 640)):
  """
  Resizes image to a smaller dimension while maintaining aspect ratio.
  """
  height, width = img.shape[:2]
  if height > width:
      new_height = target_size[1]
      new_width = int(width * (new_height / height))
  else:
      new_width = target_size[0]
      new_height = int(height * (new_width / width))
  return cv2.resize(img, (new_width, new_height))
def detect_and_mask(img):
  """
  Performs object detection with YOLO, creates a mask, and returns masked image.
  """
  results = model.predict(img, conf=0.2)  # Detect objects with 20% confidence

  # Create a black mask with the same dimensions as the image
  mask = np.uint8(img)[:, :, 0]  # Use only the first channel for grayscale mask

  for r in results:
      boxes = r.boxes
      for box in boxes:
          b = box.xyxy[0]
          x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
          # Fill the bounding box area with white in the mask
          mask[y1:y2, x1:x2] = 255  # White pixels indicate masked area

  # Apply the mask to the image (optional for visualization)
  masked_img = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)

  return masked_img, mask


def detect_green(img, mask):
  """
  Checks original image and mask to detect green pixels excluding masked areas.
  """
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # Invert the mask (black becomes white, white becomes black)
  inverted_mask = cv2.bitwise_not(mask)
  # Apply bitwise AND between inverted mask and HSV image to exclude masked areas
  masked_hsv = cv2.bitwise_and(hsv, hsv, mask=inverted_mask)
  # Detect green pixels within the masked_hsv image
  inRange = cv2.inRange(masked_hsv, green_lower, green_upper)
  return inRange



def draw_contours(img, mask, color=(0, 255, 0), thickness=2):
  """
  Draws contours (bounding boxes) around detected green areas.
  """
  # Find contours (connected green regions)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours as bounding boxes
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def main():
  # cap = cv2.VideoCapture(0)
  # cap.set(3, 640)
  # cap.set(4, 640)

  #while True:
  # _, img = cap.read()
  img1 = cv2.imread("1700717741317.jpg")
  img = preprocess_image(img1)
  # Perform masked detection
  masked_img, mask = detect_and_mask(img.copy())

  # Detect green pixels
  green_mask = detect_green(masked_img, mask)

  # Combine detections (optional: adjust logic for desired output)
  combined_mask = cv2.bitwise_or(mask, green_mask)  # Mask out previously detected objects
  combined_img = cv2.bitwise_and(img.copy(), img.copy(), mask=combined_mask)

  # Draw bounding boxes for YOLO detections (optional: modify YOLO to draw boxes)
  # ... (code to use YOLO's visualization tools or implement custom box drawing)

  # Draw contours for green detections
  draw_contours(combined_img, green_mask)

  # Display results
  cv2.imshow("Masked Detection", masked_img)
  cv2.imshow("Green Detection", green_mask)
  cv2.imshow("Combined Detection (Green Boxes)", combined_img)
  cv2.waitKey(0)
  cv2

  # if cv2.waitKey(1) & 0xFF == ord(' '):
  #     break

  # cap.release()
  # cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
