   
import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import threading
import queue


# Construct the path to the sound file
sound_file_path = os.path.join(os.path.dirname(__file__), 'sounds', 'ala.wav')
sound_eye_path = os.path.join(os.path.dirname(__file__), 'sounds', 'eye.wav')
video_file_path = os.path.join(os.path.dirname(__file__), 'sounds', 'lastvid.mp4')
file = video_file_path = os.path.join(os.path.dirname(__file__), 'can.txt')
speed_kmh = 0

"""def readFileData(filename,  stop_event):
    #Thread function to read and process data from a file.
    global speed_kmh
    try:
        
        with open(filename, 'r') as file:
            for line in file:
                
                if stop_event.is_set():
                    break
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if "Get data from ID: 0x38D" in line:
                    next_line = True
                elif next_line and stripped_line:
                    data = line.strip().split()
                    
                    
                    if len(data) >= 6:
                        byte1 = data[0]  # Extract Byte 1
                        byte5 = data[4]  # Extract Byte 2
                        speed_raw = int(byte1, 16) * 256 + int(byte5, 16)
                        speed_kmh = (speed_raw * 0.01) 
                        print(f"Reading line: {line.strip().split()}")
                        print(byte1, byte5) 
                        print(speed_kmh)
                        time.sleep(1)
                        
                        
                time.sleep(1)  # Simulate a delay as if reading from a slow serial port
    except Exception as e:
        print(f"Error in file reading thread: {e}")
        speed_kmh = 0.01 """

def play_warning_sound(sound_path,stop_event):
    while not stop_event.is_set():
        playsound(sound_path) 

def my_variables():
    with open('hand.txt', 'r') as file:
        hand = int(file.read())
        
    
    with open('face.txt', 'r') as file:
        face = int(file.read())
        
    
    return hand, face

"""def record_video(file_number):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # You can also use 'MP4V' or 'MJPG'
    switch_time = 3600 
    start_time = time.time()
    out = cv2.VideoWriter(f'output{file_number}.mp4', fourcc, 60.0, (640, 480))
    return out, switch_time, start_time"""



class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        
            
        return image    
    def positionFinder(self, image, draw=True):
        lmlist = []
        palm_centers = []
        fingertip_positions = []

        if self.results.multi_hand_landmarks:
            for handNo, hand in enumerate(self.results.multi_hand_landmarks):
                # Palm landmarks (base of the thumb, pinky, index, and middle finger)
                palm_landmarks = [0, 5, 9, 13, 17]
                fingertip_landmarks = [4, 8, 12, 16, 20]
                x_list = []
                y_list = []
                for id, lm in enumerate(hand.landmark):
                    if id in palm_landmarks:
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        x_list.append(cx)
                        y_list.append(cy)
                    if id in fingertip_landmarks:  # Fingertip landmarks
                        fingertip_positions.append((cx, cy))
                        if draw:
                            cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Calculate the center of the palm
                palm_center_x, palm_center_y = int(np.mean(x_list)), int(np.mean(y_list))
                palm_centers.append((palm_center_x, palm_center_y))

                if draw:
                    cv2.circle(image, (palm_center_x, palm_center_y), 15, (25, 0, 255), cv2.FILLED)

        return lmlist, palm_centers, fingertip_positions

class WheelDetector:
    def __init__(self):
        self.permanently_detected = False 
        self.stable_ellipse = None
        self.ellipse_detected_at = None
        self.check_interval = 10 # Start with checking every 10 seconds
        
    
    def detect_wheel(self, image,current_time):
        if self.permanently_detected:
            return self.stable_ellipse
        h, w = image.shape[:2]
        bottom_half = image[h//2:,h//2:]
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        print("Searching for new ellipse. Current time:", current_time)
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                orientation += 19.0  # Adjust based on your observation
                major_axis = max(axes)
                minor_axis = min(axes)
                aspect_ratio = major_axis / minor_axis
                
                print(f"Ellipse: Center={center}, Axes={axes}, Orientation={orientation}, Aspect Ratio={aspect_ratio}")  # Debugging output
                
                if aspect_ratio > 1.0 and major_axis > h / 4:
                    print(f"New ellipse detected: Center={center}, Axes={axes}, Orientation={orientation}")
                    corrected_center = (int(center[0])-110, int(center[1] + h//2+50))
                    corrected_axes = (max(axes)//1.3+30,max(axes)+70)
                    self.stable_ellipse = (corrected_center, corrected_axes, orientation)
                    self.permanently_detected = True
                    self.ellipse_detected_at = current_time  # Record the time of detection
                    break # Stop looking once we've found our ellipse
                    # If we have a stable ellipse, we check if enough time has passed before checking again
                
               
        return self.stable_ellipse

    def draw_stable_ellipse(self, image):
        # Draw the stored stable ellipse on the image
        if self.stable_ellipse:
            center, axes, orientation = self.stable_ellipse
            h = image.shape[0]
            
            
            cv2.ellipse(image, (center, axes, orientation), (0, 255, 0), 2)
            cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            
            
            return image
        
def check_hands_on_wheel(palm_centers, fingertip_position, wheel_circle, hands_last_detected_time, hands_off_duration=10):
    points_to_check = palm_centers + fingertip_position
    if wheel_circle is not None:
        wheel_center, wheel_axes, orientation = wheel_circle
        wheel_x, wheel_y = wheel_center
        major_axis, minor_axis = wheel_axes
        orientation_rad = np.deg2rad(orientation)
       
        for point in points_to_check:
            point_x, point_y = point
            transformed_x = (point_x - wheel_x) * np.cos(orientation_rad) + (point_y - wheel_y) * np.sin(orientation_rad)
            transformed_y = -(point_x - wheel_x) * np.sin(orientation_rad) + (point_y - wheel_y) * np.cos(orientation_rad)
            
            if (transformed_x**2 / (major_axis/2)**2 + transformed_y**2 / (minor_axis/2)**2) <= 1:
                return True, time.time()  # Reset the timer
    if hands_last_detected_time and (time.time() - hands_last_detected_time > hands_off_duration):
        return False, None
    return False, hands_last_detected_time

class DrowsinessDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh with specific settings for drowsiness and sunglasses detection
        self.drowsy_start = None
        self.drowsy_threshold = 1  # Threshold in seconds for drowsiness
        self.EAR_THRESHOLD = 0.21
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5,refine_landmarks=True )

    def calculate_ear(self, eye):
    # Compute the distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Compute the distance between the horizontal eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye[0] - eye[3])
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def check_sunglasses(self, image, eye_points):
        # Crop eye regions from the image
        eye_regions = [image[int(min(eye[:, 1])):int(max(eye[:, 1])), int(min(eye[:, 0])):int(max(eye[:, 0]))] for eye in eye_points]
        
        darkness_threshold = 20  # Threshold for average pixel intensity that might indicate sunglasses, needs tuning
    
        for region in eye_regions:
            if region.size == 0:  # Avoid division by zero
                continue
            if np.mean(region) < darkness_threshold:
                print("Average brightness in eye region:", np.mean(region))
                return True  # Likely wearing sunglasses
            
        return False

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    wheel_detector = WheelDetector()
    eye_detector = DrowsinessDetector()
    hands_last_detected_time = None
    warning_triggered = False
    warning_sound_thread = None
    sound_thread = None
    stop_sound_event = threading.Event()
    stop_eye_sound = threading.Event()
    stop_event = threading.Event()  # Event to signal the thread to stop
    #file_thread = threading.Thread(target=readFileData, args=(file,  stop_event))
    #file_thread.start()
    
    
    """file_number = 1
    record = record_video(file_number)"""""
    

    while True:
        success, image = cap.read()
        if not success:
            break
        
        """record[0].write(image)
        if (time.time() - record[2]) >= record[1]:
            Release the current VideoWriter
            record[0].release()
            file_number = 2 if file_number == 1 else 1
            record = record_video(file_number)"""
           
        ui = my_variables()    
        image = tracker.handsFinder(image)
        current_time = time.time()
        wheel_circle = wheel_detector.detect_wheel(image, current_time)
        lmlist, palm_centers,fingertip_position = tracker.positionFinder(image)
        wheel_detector.draw_stable_ellipse(image)
         # Check if hands are on the wheel and update the timer
        hands_on_wheel, updated_time = check_hands_on_wheel(palm_centers, fingertip_position, wheel_circle, hands_last_detected_time)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = eye_detector.face_mesh.process(rgb_image)
        sunglasses_detected = False
        
       
        
        if hands_on_wheel:
            
            hands_last_detected_time = updated_time
            warning_triggered = False
            if warning_sound_thread is not None:
                stop_sound_event.set()  # Stop the sound
        else:
            if hands_last_detected_time and (current_time - hands_last_detected_time > 3) and not warning_triggered:
                
                if warning_sound_thread is None or not warning_sound_thread.is_alive() and (ui[0] == 1):
                    stop_sound_event.clear()
                    warning_sound_thread = threading.Thread(target=play_warning_sound, args=(sound_file_path, stop_sound_event))
                    warning_sound_thread.start()
                warning_triggered = True
            elif not hands_last_detected_time:
                hands_last_detected_time = current_time
        if hands_on_wheel:
            cv2.putText(image, "HANDS ARE ON THE WHEEL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif warning_triggered:
            cv2.putText(image, "HANDS ARE OFF THE WHEEL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        #print("Hands on wheel:", hands_on_wheel)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lh_indices = [362, 385, 387, 263, 373, 380]  # Update with correct indices
                rh_indices = [33, 160, 158, 133, 153, 144] # Update with correct indices
                
                lh = np.array([(face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]) for i in lh_indices])
                rh = np.array([(face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]) for i in rh_indices])
            

                # Calculate EAR for both eyes
                left_ear = eye_detector.calculate_ear(lh)
                right_ear = eye_detector.calculate_ear(rh)

                # Average the EARs together for both eyes
                ear = (left_ear + right_ear) / 2.0

                # Use ear threshold to check if the person is blinking or drowsy
                EAR_THRESHOLD = 0.21  # Threshold value 
                if eye_detector.check_sunglasses(image, [lh, rh]):
                    cv2.putText(image, "SUNGLASSES DETECTED", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    sunglasses_detected = True

                if not sunglasses_detected:
                    if ear < EAR_THRESHOLD:
                        if drowsy_start is None:
                            drowsy_start = time.time()  # Start timer if not already started
                    else:
                        drowsy_start = None  # Reset timer if eyes are open

                # Check if the eyes have been closed for more than the drowsy threshold
                    if drowsy_start and (time.time() - drowsy_start) > eye_detector.drowsy_threshold:
                        cv2.putText(image, "DROWSY", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        #Start a new thread to play the warning sound
                        if sound_thread is None or not sound_thread.is_alive() and (ui[1] == 1):
                            stop_eye_sound.clear()
                            sound_thread = threading.Thread(target=play_warning_sound, args=(sound_eye_path,stop_eye_sound))
                            sound_thread.start()
                    else:
                        cv2.putText(image, "AWAKE", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if sound_thread is not None:
                
                            stop_eye_sound.set()
        
            

         

        cv2.imshow("Video", image)
        
        print("-------",speed_kmh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              # Tell the thread to stop
            stop_event.set()
            #file_thread.join()
            stop_sound_event.set()
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == "__main__":
    main()
