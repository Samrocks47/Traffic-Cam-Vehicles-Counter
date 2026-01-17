import cv2
import numpy as np
from ultralytics import YOLO
from sort import *

#1.0-------ROI Selection Function-------#
print(version:=cv2.__version__)
def roi_selection(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h, w = frame.shape[:2]

    pts = np.array([
        [w//4, h//4],
        [3*w//4, h//4],
        [3*w//4, 3*h//4],
        [w//4, 3*h//4]
    ], dtype=np.int32)

    drag_idx = -1
    R = 10

    def mouse(event, x, y, flags, param):
        nonlocal drag_idx, pts

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(pts):
                if np.hypot(x - p[0], y - p[1]) < R:
                    drag_idx = i
                    break

        elif event == cv2.EVENT_MOUSEMOVE and drag_idx != -1:
            pts[drag_idx] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drag_idx = -1

    win = "Resizable Quad ROI (ENTER=done, ESC=cancel)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)

    while True:
        temp = frame.copy()

        cv2.polylines(temp, [pts], True, (0, 255, 0), 2)
        M = pts.mean(axis=0).astype(int)
        cx, cy = M
        cv2.circle(temp, (cx, cy), 5, (0, 0, 255), -1)
        for (x, y) in pts:
            cv2.drawMarker(
                temp, (x, y),
                (0, 0, 255),
                cv2.MARKER_DIAMOND,
                20, 2
            )

        cv2.imshow(win, temp)
        k = cv2.waitKey(1) & 0xFF

        if k in (13, 10):
            break
        elif k == 27:
            return None

    cv2.destroyAllWindows()
    return pts

model= YOLO("models\\yolov8s.pt") #  yolov8l.pt yolov8n.pt yolo11m.pt yolo11x.pt
path="resources\\test3.mp4"
video=cv2.VideoCapture(path)
pts=roi_selection(path)
ret, first = video.read()
h, w = first.shape[:2]

names=[ "car", "motorbike", "bus", "truck"]
names_color={ "car": (0, 0, 255), "motorbike": (0, 0, 255), "bus": (0, 0, 255), "truck": (0, 0, 255)}
mask = np.zeros(first.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [pts], 255)
tracker = Sort(max_age=22, min_hits=3, iou_threshold=0.3)
counted_ids = set()

while True:
    ret, frame = video.read()
    if not ret:
        break

    detections = np.empty((0, 5))
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls < len(names) and names[cls] in ["car", "bus", "truck", "motorbike"]:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
    # Perspective-based counting line (65% depth)
    tracks = tracker.update(detections)
    for x1, y1, x2, y2, track_id in tracks.astype(int):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        if mask[cy, cx] == 255:
            counted_ids.add(track_id)
            color = (0,255,0)  # inside ROI
        else:
            color = (0,0,255) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        #Counting Logic
        
    #display count
    cv2.polylines(frame, [pts], True, (0,255,0), 2)
    centroid = pts.mean(axis=0).astype(int)
    cv2.circle(frame, tuple(centroid), 5, (0,0,255), -1)
    for (x, y) in pts:
        cv2.line(frame, (x, y), tuple(centroid), (255,0,0), 1)

    # show count
    cv2.putText(frame, f"COUNT: {len(counted_ids)}", (40,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    cv2.imshow("Tracking + Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
print("FINAL COUNT:", len(counted_ids))