import cv2
import numpy as np
import time
import uuid


try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise FileNotFoundError("Haar Cascade file not found or invalid.")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    exit()


def try_open_webcam(max_index=3):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam opened successfully at index {i}")
            return cap, i
        cap.release()
    return None, -1

cap, cam_index = try_open_webcam()
if cap is None:
    print("Error: No webcam found. Please check connection or try different index.")
    exit()


tracked_faces = {}
head_count = 0


DISTANCE_THRESHOLD = 80         
FORGET_AFTER_SECONDS = 1.5        
MIN_FACE_SIZE = 100               
CONFIDENCE_THRESHOLD = 1.0       
STABLE_FRAMES = 10               
FRAME_RATE = 30                   
MIN_NEW_FACE_GAP = 0.8            


prev_gray = None
prev_points = None
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
last_new_face_time = 0            


def get_centroid(x, y, w, h):
    return (x + w // 2, y + h // 2)


def calculate_confidence(w, h):
    aspect_ratio = w / h
    size_score = min(w, h) / MIN_FACE_SIZE
    aspect_score = 1.0 - abs(aspect_ratio - 1.0)
    return min(size_score * aspect_score, 1.0)


def track_with_optical_flow(gray, prev_gray, prev_points, last_box):
    if prev_points is None or prev_gray is None or len(prev_points) == 0:
        return None, None
    try:
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        if status is None or len(status) == 0 or status.sum() < len(status) * 0.5:
            return None, None
        good_points = new_points[status == 1]
        if len(good_points) == 0:
            return None, None
        new_centroid = np.mean(good_points, axis=0).astype(int)
        x, y, w, h = last_box
        shift = new_centroid - np.mean(prev_points[status == 1], axis=0).astype(int)
        new_box = (x + shift[0] - w // 2, y + shift[1] - h // 2, w, h)
        return new_box, good_points
    except Exception:
        return None, None


frame_count = 0
cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame_count += 1
    current_time = time.time()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error converting frame to grayscale: {e}")
        continue

  
    detected_faces = []
    if frame_count % 5 == 0:
        try:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.03,  
                minNeighbors=12,  
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
            for (x, y, w, h) in faces:
                confidence = calculate_confidence(w, h)
                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_faces.append((x, y, w, h, confidence))
        except Exception as e:
            print(f"Error during face detection: {e}")
            continue

    
    new_tracked_faces = {}
    for face_id, (centroid, timestamp, last_box, frame_countdown) in tracked_faces.items():
        new_box, new_points = track_with_optical_flow(gray, prev_gray, prev_points, last_box)
        if new_box:
            x, y, w, h = [int(v) for v in new_box]
            new_centroid = get_centroid(x, y, w, h)
            new_tracked_faces[face_id] = (new_centroid, current_time, (x, y, w, h), frame_countdown)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {face_id[:4]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
           
            if current_time - timestamp < FORGET_AFTER_SECONDS / 2:
                new_tracked_faces[face_id] = (centroid, timestamp, last_box, frame_countdown)

    
    unmatched_faces = []
    for (x, y, w, h, confidence) in detected_faces:
        centroid = get_centroid(x, y, w, h)
        matched = False
        closest_dist = float('inf')
        closest_id = None
        for face_id, (old_centroid, timestamp, last_box, frame_countdown) in new_tracked_faces.items():
            dist = np.linalg.norm(np.array(centroid) - np.array(old_centroid))
            if dist < DISTANCE_THRESHOLD and dist < closest_dist:
                closest_dist = dist
                closest_id = face_id
                matched = True
        if matched:
            new_tracked_faces[closest_id] = (centroid, current_time, (x, y, w, h), max(new_tracked_faces[closest_id][3], STABLE_FRAMES))
        else:
            unmatched_faces.append((x, y, w, h, confidence, centroid))

    
    for (x, y, w, h, confidence, centroid) in unmatched_faces:
        
        if (current_time - last_new_face_time > MIN_NEW_FACE_GAP and
            all(np.linalg.norm(np.array(centroid) - np.array(old_centroid)) > DISTANCE_THRESHOLD
                for old_centroid, _, _, _ in new_tracked_faces.values())):
            face_id = str(uuid.uuid4())
            new_tracked_faces[face_id] = (centroid, current_time, (x, y, w, h), STABLE_FRAMES)
            if frame_count > STABLE_FRAMES * 2:
                head_count += 1
                last_new_face_time = current_time
                print(f"New face counted: ID {face_id[:4]}, Total count: {head_count}")

    # Update tracked faces
    tracked_faces = {
        face_id: (
            centroid,
            timestamp,
            box,
            max(0, frame_countdown - 1) if frame_countdown > 0 else 0
        )
        for face_id, (centroid, timestamp, box, frame_countdown) in new_tracked_faces.items()
        if current_time - timestamp < FORGET_AFTER_SECONDS
    }

    # Update optical flow variables
    prev_gray = gray.copy()
    prev_points = None
    if detected_faces:
        x, y, w, h, _ = detected_faces[0]
        prev_points = np.float32([[[x + w // 2, y + h // 2]]])

    # Display stats
    cv2.putText(frame, f"Unique Faces: {head_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    try:
        cv2.imshow("Face Counter", frame)
    except Exception as e:
        print(f"Error displaying frame: {e}")
        break

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Total unique faces detected: {head_count}")