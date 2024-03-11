import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("D:/Games/best.pt")

green_lower = np.array([25, 50, 50])
green_upper = np.array([95, 255, 255])
def detect_and_mask(img):
  results = model.predict(img, conf=0.2)

  mask = np.uint8(img)[:, :, 0]

  for r in results:
      boxes = r.boxes
      for box in boxes:
          b = box.xyxy[0]
          x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
          mask[y1:y2, x1:x2] = 255

  masked_img = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)

  return masked_img, mask
def detect_green(img, mask):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  inverted_mask = cv2.bitwise_not(mask)
  masked_hsv = cv2.bitwise_and(hsv, hsv, mask=inverted_mask)
  inRange = cv2.inRange(masked_hsv, green_lower, green_upper)
  return inRange
def draw_contours(img, mask, color=(0, 255, 0), thickness=2):
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
      # cv2.putText(img, "Weeds", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
def main():
  cap = cv2.VideoCapture(0)
  cap.set(3, 640)
  cap.set(4, 640)

  while True:
      _, img = cap.read()
      masked_img, mask = detect_and_mask(img.copy())
      green_mask = detect_green(masked_img, mask)

      combined_mask = cv2.bitwise_or(mask, green_mask)
      combined_img = cv2.bitwise_and(img.copy(), img.copy(), mask=combined_mask)

      draw_contours(combined_img, green_mask)

      cv2.imshow("1", masked_img)
      cv2.imshow("2", green_mask)
      cv2.imshow("3", combined_img)
      if cv2.waitKey(1) & 0xFF == ord(' '):
          break
  cap.release()
  cv2.destroyAllWindows()
if __name__ == "__main__":
  main()
