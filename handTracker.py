import cv2
import mediapipe as mp
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
    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        a,b,c,d = 0,0,0,0
        
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                if id == 0:
                    a,b = cx,cy
                if id == 5:
                    c,d = cx,cy
              
                fx,fy = int((a+c)),int((b+d)/2)
                print(fx,fy)
            cv2.circle(image,(fx,fy), 15 , (25,0,255), cv2.FILLED)
            #if draw:
                #cv2.circle(image,(cx,cy), 5 , (255,0,255), cv2.FILLED)

        return lmlist
def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            print(lmList[5])
            print(lmList[0])
            print(lmList[17])
        
        cv2.imshow("Video",image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    main()