import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/shapes_detector/weights/best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam!")
    exit()

print("🎥 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # === Display results with bounding boxes ===
    annotated_frame = results[0].plot()
    cv2.imshow("Shape Detection (Press 'q' to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
