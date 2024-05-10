import tkinter as tk

class UI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Driving Assistance System")
        self.geometry("350x500")

        self.layout = tk.Frame(self)
        self.layout.pack()

        self.lane_var = tk.IntVar()
        self.lane_checkbox = tk.Checkbutton(self.layout, text="Lane Detection", variable=self.lane_var, command=self.save_data)
        self.lane_checkbox.config(font=("Arial", 14))
        self.lane_checkbox.pack()

        self.face_var = tk.IntVar()
        self.face_checkbox = tk.Checkbutton(self.layout, text="Face", variable=self.face_var, command=self.save_data)
        self.face_checkbox.config(font=("Arial", 14))
        self.face_checkbox.pack()

        self.hand_var = tk.IntVar()
        self.hand_checkbox = tk.Checkbutton(self.layout, text="Hand", variable=self.hand_var, command=self.save_data)
        self.hand_checkbox.config(font=("Arial", 14))
        self.hand_checkbox.pack()

        self.traffic_lights_var = tk.IntVar()
        self.traffic_lights_checkbox = tk.Checkbutton(self.layout, text="Traffic lights", variable=self.traffic_lights_var, command=self.save_data)
        self.traffic_lights_checkbox.config(font=("Arial", 14))
        self.traffic_lights_checkbox.pack()

        self.traffic_signs_var = tk.IntVar()
        self.traffic_signs_checkbox = tk.Checkbutton(self.layout, text="Traffic signs", variable=self.traffic_signs_var, command=self.save_data)
        self.traffic_signs_checkbox.config(font=("Arial", 14))
        self.traffic_signs_checkbox.pack()

    def save_data(self):
        data = {
            'lane': "1" if self.lane_var.get() else "0",
            'face': "1" if self.face_var.get() else "0",
            'hand': "1" if self.hand_var.get() else "0",
            'traffic_lights': "1" if self.traffic_lights_var.get() else "0",
            'traffic_signs': "1" if self.traffic_signs_var.get() else "0"
        }

        for key, value in data.items():
            with open(f"{key}.txt", 'w') as file:
                file.write(value)

if __name__ == "__main__":
    collector = UI()
    collector.mainloop()
