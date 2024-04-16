   
import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import threading


# Construct the path to the sound file
sound_file_path = os.path.join(os.path.dirname(__file__), 'sounds', 'ala.wav')
video_file_path = os.path.join(os.path.dirname(__file__), 'sounds', 'vid.mp4')
def play_warning_sound(sound_path,stop_event):
    while not stop_event.is_set():
        playsound(sound_path) 

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
        bottom_half = image[h//2:, :]
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
                orientation += 12.0  # Adjust based on your observation
                major_axis = max(axes)
                minor_axis = min(axes)
                aspect_ratio = major_axis / minor_axis
                
                print(f"Ellipse: Center={center}, Axes={axes}, Orientation={orientation}, Aspect Ratio={aspect_ratio}")  # Debugging output
                
                if aspect_ratio > 1.0 and major_axis > h / 4:
                    print(f"New ellipse detected: Center={center}, Axes={axes}, Orientation={orientation}")
                    corrected_center = (int(center[0])-10, int(center[1] + h//2+40))
                    corrected_axes = (max(axes)//1.2,max(axes))
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

def main():
    cap = cv2.VideoCapture(video_file_path)
    tracker = handTracker()
    wheel_detector = WheelDetector()
    hands_last_detected_time = None
    warning_triggered = False
    warning_sound_thread = None
    stop_sound_event = threading.Event()
    

    while True:
        success, image = cap.read()
        if not success:
            break

        image = tracker.handsFinder(image)
        current_time = time.time()
        wheel_circle = wheel_detector.detect_wheel(image, current_time)
        lmlist, palm_centers,fingertip_position = tracker.positionFinder(image)
        wheel_detector.draw_stable_ellipse(image)
         # Check if hands are on the wheel and update the timer
        hands_on_wheel, updated_time = check_hands_on_wheel(palm_centers, fingertip_position, wheel_circle, hands_last_detected_time)
        if hands_on_wheel:
            
            hands_last_detected_time = updated_time
            warning_triggered = False
            if warning_sound_thread is not None:
                stop_sound_event.set()  # Stop the sound
        else:
            if hands_last_detected_time and (current_time - hands_last_detected_time > 3) and not warning_triggered:
                
                if warning_sound_thread is None or not warning_sound_thread.is_alive():
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
        
        print("Hands on wheel:", hands_on_wheel)
        

        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_sound_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == "__main__":
    main()
