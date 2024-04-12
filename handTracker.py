import cv2
import mediapipe as mp
import numpy as np
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

def detect_wheel(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using Histogram Equalization
    equalized_gray = cv2.equalizeHist(gray)
    
    # Apply a median blur to reduce noise while keeping edges sharp
    blurred_gray = cv2.medianBlur(equalized_gray, 5)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=120,
                               param1=100, param2=40, minRadius=75, maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        return circles[0][0]  # Assuming the first detected circle is the wheel
    return None
def check_hands_on_wheel(palm_centers, wheel_circle):
    if wheel_circle is not None:
        (wheel_x, wheel_y, wheel_r) = wheel_circle
        for (palm_x, palm_y) in palm_centers:
            if (palm_x - wheel_x) ** 2 + (palm_y - wheel_y) ** 2 < wheel_r ** 2:
                # Palm is on the wheel
                return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    
    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        wheel_circle = detect_wheel(image)

        lmList, palm_centers = tracker.positionFinder(image)
        for center in palm_centers:
            print(center)
        # Check if hands are on the wheel
        
        hands_on_wheel = check_hands_on_wheel(palm_centers, wheel_circle)
        print("Hands on wheel:", hands_on_wheel)

        cv2.imshow("Video", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
