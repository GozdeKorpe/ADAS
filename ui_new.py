import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont  # QFont eklendi

class UI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Driving Assistance System")
        self.setFixedSize(350, 500)  # Pencere boyutunu ayarla

        self.layout = QVBoxLayout()

        self.lane_detection_checkbox = QCheckBox("Lane Detection")
        self.lane_detection_checkbox.stateChanged.connect(self.save_data)
        self.lane_detection_checkbox.setFont(QFont("Arial", 40))  # Font boyutu ve stili ayarlandı
        self.layout.addWidget(self.lane_detection_checkbox)
        
        self.face_checkbox = QCheckBox("Face")
        self.face_checkbox.stateChanged.connect(self.save_data)
        self.face_checkbox.setFont(QFont("Arial", 40))  # Font boyutu ve stili ayarlandı
        self.layout.addWidget(self.face_checkbox)

        self.hand_checkbox = QCheckBox("Hand")
        self.hand_checkbox.stateChanged.connect(self.save_data)
        self.hand_checkbox.setFont(QFont("Arial", 40))  # Font boyutu ve stili ayarlandı
        self.layout.addWidget(self.hand_checkbox)
        
        self.traffic_lights_checkbox = QCheckBox("Traffic lights")
        self.traffic_lights_checkbox.stateChanged.connect(self.save_data)
        self.traffic_lights_checkbox.setFont(QFont("Arial", 40))  # Font boyutu ve stili ayarlandı
        self.layout.addWidget(self.traffic_lights_checkbox)
        
        self.traffic_signs_checkbox = QCheckBox("Traffic signs")
        self.traffic_signs_checkbox.stateChanged.connect(self.save_data)
        self.traffic_signs_checkbox.setFont(QFont("Arial", 40))  # Font boyutu ve stili ayarlandı
        self.layout.addWidget(self.traffic_signs_checkbox)

        self.setLayout(self.layout)

    def save_data(self):
        data = {
            'lane': "1" if self.lane_detection_checkbox.isChecked() else "0",
            'face': "1" if self.face_checkbox.isChecked() else "0",
            'hand': "1" if self.hand_checkbox.isChecked() else "0",
            'traffic_lights': "1" if self.traffic_lights_checkbox.isChecked() else "0",
            'traffic_signs': "1" if self.traffic_signs_checkbox.isChecked() else "0"
        }

        for key, value in data.items():
            with open(f"{key}.txt", 'w') as file:
                file.write(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    collector = UI()
    collector.show()
    sys.exit(app.exec_())
