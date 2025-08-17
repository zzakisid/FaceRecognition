import cv2
import dlib
import numpy as np
import mediapipe as mp
import os

# Load models
face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load known faces
def load_known_faces(folder):
    known_encodings = []
    names = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Warning: Could not read image '{file}'. Skipping.")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = rgb.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            face = dlib.rectangle(x1, y1, x2, y2)
            shape = predictor(rgb, face)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            known_encodings.append(encoding)
            names.append(os.path.splitext(file)[0])
            print(f"✅ Loaded face encoding for '{file}'")
        else:
            print(f"❌ No face detected in '{file}'. Skipping.")
    return known_encodings, names

known_encodings, known_names = load_known_faces("known_faces")

# Function to run live face recognition
def run_live_recognition():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
                face_rect = dlib.rectangle(x1, y1, x2, y2)

                try:
                    shape = predictor(rgb, face_rect)
                    encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))

                    distances = np.linalg.norm(known_encodings - encoding, axis=1)
                    match_index = np.argmin(distances)
                    name = known_names[match_index] if distances[match_index] < 0.6 else "Unknown"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                except:
                    continue

        cv2.imshow("Face Recognition - Live", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
     run_live_recognition()