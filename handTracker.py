   
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

        if self.results.multi_hand_landmarks:
            for handNo, hand in enumerate(self.results.multi_hand_landmarks):
                # Palm landmarks (base of the thumb, pinky, index, and middle finger)
                palm_landmarks = [0, 5, 9, 13, 17]
                x_list = []
                y_list = []
                for id, lm in enumerate(hand.landmark):
                    if id in palm_landmarks:
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        x_list.append(cx)
                        y_list.append(cy)
                        if draw:
                            cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Calculate the center of the palm
                palm_center_x, palm_center_y = int(np.mean(x_list)), int(np.mean(y_list))
                palm_centers.append((palm_center_x, palm_center_y))

                if draw:
                    cv2.circle(image, (palm_center_x, palm_center_y), 15, (25, 0, 255), cv2.FILLED)

        return lmlist, palm_centers

class WheelDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.stable_circle_coordinates = None
        self.circle_detected_at = None
        self.check_circle_interval = 0.2  # Start with checking every 0.1 seconds
        
    
    def detect_wheel(self, image, current_time):
        if current_time - self.last_detection_time > self.check_circle_interval:
            # Focus on the bottom half of the image
            h, w = image.shape[:2]
            bottom_half = image[h//2:, :]
            gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                       param1=100, param2=30, minRadius=75, maxRadius=300)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Assuming the first detected circle is the wheel
                self.last_detection_time = current_time
                if self.stable_circle_coordinates is None:
                    self.stable_circle_coordinates = circles[0][0]
                if self.circle_detected_at is None:
                    self.circle_detected_at = current_time
                elif current_time - self.circle_detected_at >= 30:
                    # Once the circle has been stable for 10 seconds, check every 1 minute
                    self.check_circle_interval = 40
                return circles[0][0]
            else:
                self.circle_detected_at = None  # Reset if no circle is found
        return self.stable_circle_coordinates
    def draw_stable_circle(self, image):
        """Draws the stable circle on the image if the coordinates are available."""
        if self.stable_circle_coordinates is not None:
            # Unpack the circle coordinates and radius
            x, y, r = self.stable_circle_coordinates
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

def check_hands_on_wheel(palm_centers, wheel_circle, hands_last_detected_time, hands_off_duration=10):
    if wheel_circle is not None:
        wheel_x, wheel_y, wheel_radius = wheel_circle
        for palm_center in palm_centers:
            palm_x, palm_y = palm_center
            if np.sqrt((palm_x - wheel_x) ** 2 + (palm_y - wheel_y) ** 2) <= wheel_radius:
                return True, time.time()  # Reset the timer
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
        lmlist, palm_centers = tracker.positionFinder(image)
        wheel_detector.draw_stable_circle(image)
         # Check if hands are on the wheel and update the timer
        hands_on_wheel, updated_time = check_hands_on_wheel(palm_centers, wheel_circle, hands_last_detected_time)
        if hands_on_wheel:
            hands_last_detected_time = updated_time
            warning_triggered = False
            if warning_sound_thread is not None:
                stop_sound_event.set()  # Stop the sound
        else:
            if hands_last_detected_time and (current_time - hands_last_detected_time > 3) and not warning_triggered:
                cv2.putText(image, "HANDS ARE OFF THE WHEEL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if warning_sound_thread is None or not warning_sound_thread.is_alive():
                    stop_sound_event.clear()
                    warning_sound_thread = threading.Thread(target=play_warning_sound, args=(sound_file_path, stop_sound_event))
                    warning_sound_thread.start()
                warning_triggered = True
            elif not hands_last_detected_time:
                hands_last_detected_time = current_time
        
        print("Hands on wheel:", hands_on_wheel)
        

        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_sound_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == "__main__":
    main()
