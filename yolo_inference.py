import cv2
from ultralytics import YOLO

model = YOLO("models/football_detection.pt")
cap = cv2.VideoCapture("input_videos/ars_vs_wol.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Football_Analysis", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
