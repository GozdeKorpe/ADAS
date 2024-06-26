import cv2
import numpy as np
import time
import threading
import os
import serial
import playsound
import pygame
import sounddevice as sd
import simpleaudio as sa

file = os.path.join(os.path.dirname(__file__), 'can.txt')
lanetxt = os.path.join(os.path.dirname(__file__), 'lane.txt')
sound_file_path = os.path.join(os.path.dirname(__file__), '/home/jetson/Desktop/ADAS/sounds/alert.wav')
video_file_path = os.path.join(os.path.dirname(__file__),'/home/jetson/inference/jetson-inference/data/networks/sign/video.mp4')
#ser = serial.Serial('COM11', 750)  # Adjust COM port as necessary
#time.sleep(2)  
#ser.flushInput()

def play_sound():
    wave_obj = sa.WaveObject.from_wave_file(sound_file_path)
    play_obj = wave_obj.play()
    #play_obj.wait_done()

play_obj = play_sound()
        
def record_video(file_number):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Use 'mp4v' codec
    switch_time = 300  # When to switch each video (in seconds)
    start_time = time.time()
    out = cv2.VideoWriter(f'output{file_number}.mp4', fourcc, 30.0, (1280, 720))
    return out, switch_time, start_time

def my_variables():
    try:
        with open(lanetxt, 'r') as file:
            lane = int(file.read())
            print("Lane ->", lane)
    except ValueError:
        print("Value not found in the file.")
        lane = None
    return lane
def readFileData(filename,  stop_event):
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
                        
                        
                        
                time.sleep(1)  # Simulate a delay as if reading from a slow serial port
    except Exception as e:
        print(f"Error in file reading thread: {e}")
        speed_kmh = 0.01

def canData(ser,  stop_event):
    #Thread function to read and process data from a file.
    global speed_kmh
    try:
        
        while not stop_event.is_set():
                
            if stop_event.is_set():
                break
            line = ser.readline().decode().strip()
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
                        
    except Exception as e:
        print(f"Error in reading thread: {e}")
        speed_kmh = 0.01

class LaneDetector:
    def __init__(self):
        self.prev_left_avg_line = None
        self.prev_right_avg_line = None
        self.sol_flag = False
        self.sag_flag = False
        self.car_r_point = (1183, 809)  #arabanin sag noktasi
        self.car_l_point = (614, 809)    #arabanin sol noktasi
        self.sol_flag = False
        self.sag_flag = False




    def find_lane_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]
        roi_height = int(height * 0.6)
        roi_width = int(width * 0.6)
        gap = 210
        roi = gray[roi_height:height, gap:width - gap-150]

        edges = cv2.Canny(roi, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
        self.car_m_point = (int(width/2)-70,870)
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

        left_avg_line = np.mean(left_lines, axis=0, dtype=np.int32) if left_lines else self.prev_left_avg_line
        right_avg_line = np.mean(right_lines, axis=0, dtype=np.int32) if right_lines else self.prev_right_avg_line
        
        if left_avg_line is not None and right_avg_line is not None:
            left_x1, left_y1, left_x2, left_y2 = left_avg_line
            right_x1, right_y1, right_x2, right_y2 = right_avg_line
            
            # Draw lanes based on default values while drawing lanes
            if self.sag_flag == False and self.sol_flag == False and my_variables() == 1:
                
                #cv2.circle(img, (25, 460), 10, (0, 255, 0), 1)
                cv2.line(img, (left_x1 + gap, left_y1 + roi_height), (left_x2 + gap, left_y2 + roi_height), (0, 255, 255), 3)
                cv2.line(img, (right_x1 + gap, right_y1 + roi_height), (right_x2 + gap, right_y2 + roi_height), (0, 255, 255), 3)
                cv2.circle(img, (left_x1 + gap, left_y1 + roi_height), 10, (0, 255, 0), 1)
                cv2.circle(img, (right_x2 + gap, right_y2 + roi_height), 10, (0, 255, 0), 1)
                
            # Fill the area between lanes with green
            if self.sag_flag == False and self.sol_flag == False and (my_variables() == 1):
                print("lane girdiiiii")
                pts = np.array([[right_x1 + gap, right_y1 + roi_height],
                            [right_x2 + gap, right_y2 + roi_height],
                            [left_x1 + gap, left_y1 + roi_height], 
                            [left_x2 + gap, left_y2 + roi_height]], np.int32)
                cv2.fillPoly(img, [pts], (158, 125, 99))

        
        cv2.imshow('frame', img)

        return img, left_avg_line, right_avg_line

    def detect_crossing(self, current_left_avg_line, current_right_avg_line):
        if self.prev_left_avg_line is not None and self.prev_right_avg_line is not None:
            my_lx1 = current_left_avg_line[0]
            my_lx2 = current_left_avg_line[2]
            
            my_rx1 = current_right_avg_line[0]
            my_rx2 = current_right_avg_line[2]
            
            
            if (my_rx2 + 210 - self.car_m_point[0]) < 100:  #210 gap
                if not self.sol_flag and (my_variables() == 1):
                    print("saga geçildi")
                    self.sag_flag = True
                    play_sound()
                    #threading.Thread(target = play_warning_sound).start()
                                   
            if (my_rx2 + 210 - self.car_m_point[0]) > 400:
                if self.sag_flag:
                    print("Düzeldi")
                    self.sag_flag = False


                     
            if (self.car_m_point[0] - (my_lx1 +210) ) < 120:  #210 gap
                if not self.sol_flag and (my_variables() == 1):
                    print("sola geçildi")
                    self.sol_flag = True 
                    play_sound()
                    #threading.Thread(target = play_warning_sound).start()

                        
                     
            if (self.car_m_point[0] - (my_lx1 + 210) ) > 120:
                if self.sol_flag:
                    print("Düzeldi")
                    self.sol_flag = False

                     

def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cap.read()
    stop_sound = threading.Event()
    file_number = 1
    record = record_video(1)
    record_processed = record_video(3)
    lane_detector = LaneDetector()  # Create an instance of LaneDetector class

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        record[0].write(frame)
        """if (time.time() - record[2]) >= record[1]:
            # Release the current VideoWriter
            record[0].release()
            file_number = 2 if file_number == 1 else 1
            record = record_video(file_number)"""
        
        frame_with_lanes, current_left_avg_line, current_right_avg_line = lane_detector.find_lane_lines(frame)
        if current_left_avg_line is not None and current_right_avg_line is not None:
            lane_detector.detect_crossing(current_left_avg_line, current_right_avg_line)
            lane_detector.prev_left_avg_line = current_left_avg_line
            lane_detector.prev_right_avg_line = current_right_avg_line
        record_processed[0].write(frame_with_lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    record[0].release()


# Example usage
if __name__ == "__main__":
    main()
