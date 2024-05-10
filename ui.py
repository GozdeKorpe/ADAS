import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QCheckBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class LaneDetection:
    def __init__(self):
        self.prev_left_avg_line = None
        self.prev_right_avg_line = None
        self.sol_flag = False  # Initialize sol_flag here

    def find_lane_lines(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = img.shape[:2]
        roi_height = int(height * 0.6)
        roi_width = int(width * 0.6)
        gap = int(width * 0)
        roi = gray[roi_height:height, gap:width - gap]

        edges = cv2.Canny(roi, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)

        left_lines = []
        right_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > 0.3:
                        if slope < 0:
                            left_lines.append(line[0])
                        else:
                            right_lines.append(line[0])

        # Ortalama çizgiyi bul
        left_avg_line = np.mean(left_lines, axis=0, dtype=np.int32) if left_lines else self.prev_left_avg_line
        right_avg_line = np.mean(right_lines, axis=0, dtype=np.int32) if right_lines else self.prev_right_avg_line

        # Eğer sol ve sağ çizgi algılanırsa
        if left_avg_line is not None and right_avg_line is not None:
            left_x1, left_y1, left_x2, left_y2 = left_avg_line
            right_x1, right_y1, right_x2, right_y2 = right_avg_line

            cv2.line(img, (left_x1 + gap, left_y1 + roi_height), (left_x2 + gap, left_y2 + roi_height), (0, 255, 255), 3)
            cv2.line(img, (right_x1 + gap, right_y1 + roi_height), (right_x2 + gap, right_y2 + roi_height), (0, 255, 255), 3)

            # Alanı doldur
            pts = np.array([[right_x1 + gap, right_y1 + roi_height],
                            [right_x2 + gap, right_y2 + roi_height],
                            [left_x1 + gap, left_y1 + roi_height], 
                            [left_x2 + gap, left_y2 + roi_height]], np.int32)
            cv2.fillPoly(img, [pts], (0, 255, 0))

        # Çizgi kalınlığı ve boyutlarını ayarla
        line_thickness = 4  
        line_length = width // 6
        line_start_x = (width - line_length) // 2
        line_end_x = line_start_x + line_length
        cv2.line(img, (line_start_x, roi_height), (line_end_x, roi_height), (0, 0, 255), line_thickness)
        cv2.line(img, (450, 500), (50, 40), (0, 0, 255), line_thickness)


        # ROI sınırlarını güncelle
        if left_avg_line is not None and right_avg_line is not None:
            max_y = max(left_y1, left_y2, right_y1, right_y2) + roi_height
            min_y = min(left_y1, left_y2, right_y1, right_y2) + roi_height
            roi_height = min_y
            height = max_y


        if left_avg_line is not None and right_avg_line is not None:
            self.detect_crossing(left_avg_line, right_avg_line)
            self.prev_left_avg_line = left_avg_line
            self.prev_right_avg_line = right_avg_line
            
        roi = gray[roi_height:height, gap:width - gap]

        self.prev_left_avg_line = left_avg_line
        self.prev_right_avg_line = right_avg_line

        return img

    def detect_crossing(self, current_left_avg_line, current_right_avg_line):
        if self.prev_left_avg_line is not None and self.prev_right_avg_line is not None:
            my_lx1 = current_left_avg_line[0]
            my_lx2 = current_left_avg_line[2]
            
            my_rx1 = current_right_avg_line[0]
            my_rx2 = current_right_avg_line[2]
                
            if my_lx1 <= 30 and my_lx2 != 0 and my_rx2 > 1000:
                if not self.sol_flag:
                    print("Sola geçildi")
                    print("rx1:", my_rx1)
                    print("rx2:", my_rx2)
                    self.sol_flag = True  
            if my_lx1 >= 200 and my_lx2 != 0:
                if self.sol_flag:
                    print("Düzeldi")
                    print("rx1:", my_rx1)
                    print("rx2:", my_rx2)
                    self.sol_flag = False 


class EdaWrite:
    @staticmethod
    def edaa(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'EDAAA', (200, 200), font, 5, (255, 0, 0), 2, cv2.LINE_AA)

        return img

class VideoPlayer(QMainWindow):

    def __init__(self, video_file):
        super().__init__()
        
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)
        self.timer = QTimer()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.lane_detection = LaneDetection()
        self.eda_write = EdaWrite()

        self.lane_detection_checkbox = QCheckBox("Şerit Takibi")
        self.layout.addWidget(self.lane_detection_checkbox)

        self.eda_checkbox = QCheckBox("Ekrana Yazma")
        self.layout.addWidget(self.eda_checkbox)
        
        self.face_checkbox = QCheckBox("face")
        self.layout.addWidget(self.face_checkbox)
        
        self.hand_checkbox = QCheckBox("hand")
        self.layout.addWidget(self.hand_checkbox)
        
        self.traffic_lights_checkbox = QCheckBox("traffic lights")
        self.layout.addWidget(self.traffic_lights_checkbox)
        
        self.traffic_signs_checkbox = QCheckBox("traffic signs")
        self.layout.addWidget(self.traffic_signs_checkbox)
        
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        self.start_button = QPushButton("Başlat")
        self.start_button.clicked.connect(self.start_video)
        self.layout.addWidget(self.start_button)
        self.lane_detection = LaneDetection()
        self.timer = QTimer()

        self.timer.timeout.connect(self.update_frame)
    def start_video(self):
        self.cap = cv2.VideoCapture(self.video_file)
        self.timer.start(30)
        self.start_button.setEnabled(False)
    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        if self.lane_detection_checkbox.isChecked():
            frame = self.lane_detection.find_lane_lines(frame)
            # Yerine bu şekilde çağırıyoruz: self.lane_detection.detect_crossing()

        if self.eda_checkbox.isChecked():
            frame = self.eda_write.edaa(frame)
        
        if self.hand_checkbox.isChecked():
            hand = "1"
            print("hand 1 oldu canim")
        else:
             hand = "0"
             print("hand 0 oldu canim")

        with open('hand.txt', 'w') as file:
            file.write(hand)
               
        if self.face_checkbox.isChecked():
            face = "1"
            print("face 1 oldu ")
        else:
            face = "0"
            print("face 0 oldu")
        with open('face.txt', 'w') as file:
            file.write(face)

        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))
    

        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_file = '/Users/edanurpotuk/Desktop/gui/ADAS-main/g.mp4'
    player = VideoPlayer(video_file)
    player.setGeometry(100, 100, 800, 600)
    player.show()
    sys.exit(app.exec_())

