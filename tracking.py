import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort

if __name__ == '__main__':
    # Cargamos video o streaming
    cap = cv2.VideoCapture("videos/traffic.mp4")
    # Cargamos el modelo de YoloV8
    model = YOLO("yolov8n.pt")

    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        #results = model(frame, stream=True, verbose=False, classes=[2,3,5,7])
        results = model.track(frame, conf=0.7, iou=0.5, persist=True, verbose=False, classes=[2,3,5,7])

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            for xmin, ymin, xmax, ymax, track_id in tracks:
                """
                Analizar quien es la mejor
                """
                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        cv2.imshow("LPR GAO v.1.00", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()