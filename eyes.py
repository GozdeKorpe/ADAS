import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import threading

# Construct the path to the sound file
sound_file_path = os.path.join(os.path.dirname(__file__), 'sounds', 'ala.wav')
video_file_path = os.path.join(os.path.dirname(__file__), 'sounds', 'eye_vid.mp4')
pic = os.path.join(os.path.dirname(__file__), 'sounds', 'drowsy.jpg')
def play_warning_sound(sound_path):
    playsound(sound_path)

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True ) # This helps with accuracy when wearing glasses

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    # Compute the distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def check_sunglasses(frame, eye_points):
    # Crop eye regions from the frame
    eye_regions = [frame[int(min(eye[:, 1])):int(max(eye[:, 1])), int(min(eye[:, 0])):int(max(eye[:, 0]))] for eye in eye_points]
    
    darkness_threshold = 45  # Threshold for average pixel intensity that might indicate sunglasses, needs tuning
   
    for region in eye_regions:
        if region.size == 0:  # Avoid division by zero
            continue
        if np.mean(region) < darkness_threshold:
            print("Average brightness in eye region:", np.mean(region))
            return True  # Likely wearing sunglasses
        
    return False

# Start capturing video from the webcam
cap = cv2.imread(video_file_path)

drowsy_start = None  # Timer to start counting when eyes close
drowsy_threshold = 1  # Threshold in seconds for drowsiness


# Indices for the left and right eye in the MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB and process it with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    sunglasses_detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lh_indices = [362, 385, 387, 263, 373, 380]  # Update with correct indices
            rh_indices = [33, 160, 158, 133, 153, 144] # Update with correct indices
            
            lh = np.array([(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in lh_indices])
            rh = np.array([(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in rh_indices])
           

            # Calculate EAR for both eyes
            left_ear = calculate_ear(lh)
            right_ear = calculate_ear(rh)

            # Average the EARs together for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Use ear threshold to check if the person is blinking or drowsy
            EAR_THRESHOLD = 0.21  # Threshold value 
            if check_sunglasses(frame, [lh, rh]):
                cv2.putText(frame, "SUNGLASSES DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                sunglasses_detected = True

            if not sunglasses_detected:
                if ear < EAR_THRESHOLD:
                    if drowsy_start is None:
                        drowsy_start = time.time()  # Start timer if not already started
                else:
                    drowsy_start = None  # Reset timer if eyes are open

            # Check if the eyes have been closed for more than the drowsy threshold
                if drowsy_start and (time.time() - drowsy_start) > drowsy_threshold:
                    cv2.putText(frame, "DROWSY", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                     #Start a new thread to play the warning sound
                    sound_thread = threading.Thread(target=play_warning_sound, args=(sound_file_path,))
                    sound_thread.start()
                else:
                    cv2.putText(frame, "AWAKE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
