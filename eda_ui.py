import tkinter as tk
from PIL import Image, ImageTk

class UI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Driving Assistance System")
        self.geometry("900x800")
        #self.attributes('-fullscreen', True)  # Fullscreen mode
        self.config(bg= "light blue")
        hacettepe_logo = Image.open("download.png")
        hacettepe_logo = hacettepe_logo.resize((100, 146), resample=Image.LANCZOS)
        self.hacettepe_photo = ImageTk.PhotoImage(hacettepe_logo)

        hacettepe_label = tk.Label(self, image=self.hacettepe_photo, bg="light blue")
        hacettepe_label.pack(side=tk.TOP, pady=10, padx=400, anchor=tk.W)
        
        title_label = tk.Label(self, text="Department of Electrical and Electronics Engineering",bg= "light blue", font=("Arial", 15, "bold"), fg="black")
        title_label.pack(pady=0)
        
        title_label = tk.Label(self, text="Driving Assistance System",bg= "light blue", font=("Arial", 15, "bold"), fg="black")
        title_label.pack(pady=0)
        
        
        title_label = tk.Label(self, text="B. Aydemir, E.Potuk, G. Körpe, Y.B. Avşar",bg= "light blue", font=("Arial", 12, "bold"), fg="dark green")
        title_label.pack(pady=0)
        # Title label
        title_label = tk.Label(self, text="DRIVING ASSISTANCE SYSTEM UI",bg= "light blue", font=("Arial", 33, "bold"), fg="blue")
        title_label.pack(pady=0)
        
        self.layout = tk.Frame(self, bg= "light blue")
        self.layout.pack()

        self.lane_var = tk.IntVar()
        self.lane_checkbox = tk.Checkbutton(self.layout, text="Lane Detection", variable=self.lane_var, command=self.save_data)
        self.lane_checkbox.config(bg ="light blue" ,font=("Arial", 30), fg="black")
        self.lane_checkbox.grid(row=0, column=0, sticky=tk.W, padx=50, pady=0)
        self.lane_checkbox.bind("<Button-1>", lambda event, checkbox=self.lane_checkbox: self.change_text_color(checkbox))

        self.face_var = tk.IntVar()
        self.face_checkbox = tk.Checkbutton(self.layout, text="Fatigue Detection", variable=self.face_var, command=self.save_data)
        self.face_checkbox.config(bg ="light blue",font=("Arial", 30), fg="black")
        self.face_checkbox.grid(row=1, column=0, sticky=tk.W, padx=50, pady=8)
        self.face_checkbox.bind("<Button-1>", lambda event, checkbox=self.face_checkbox: self.change_text_color(checkbox))

        self.hand_var = tk.IntVar()
        self.hand_checkbox = tk.Checkbutton(self.layout, text="Hands Off Detection", variable=self.hand_var, command=self.save_data)
        self.hand_checkbox.config(bg ="light blue",font=("Arial", 30), fg="black")
        self.hand_checkbox.grid(row=2, column=0, sticky=tk.W, padx=50, pady=8)
        self.hand_checkbox.bind("<Button-1>", lambda event, checkbox=self.hand_checkbox: self.change_text_color(checkbox))

        self.traffic_lights_var = tk.IntVar()
        self.traffic_lights_checkbox = tk.Checkbutton(self.layout, text="Traffic Lights Detection", variable=self.traffic_lights_var, command=self.save_data)
        self.traffic_lights_checkbox.config(bg ="light blue",font=("Arial", 30), fg="black")
        self.traffic_lights_checkbox.grid(row=3, column=0, sticky=tk.W, padx=50, pady=8)
        self.traffic_lights_checkbox.bind("<Button-1>", lambda event, checkbox=self.traffic_lights_checkbox: self.change_text_color(checkbox))

        self.traffic_signs_var = tk.IntVar()
        self.traffic_signs_checkbox = tk.Checkbutton(self.layout, text="Traffic Signs Detection", variable=self.traffic_signs_var, command=self.save_data)
        self.traffic_signs_checkbox.config(bg ="light blue",font=("Arial", 30), fg="black")
        self.traffic_signs_checkbox.grid(row=4, column=0, sticky=tk.W, padx=50, pady=8)
        self.traffic_signs_checkbox.bind("<Button-1>", lambda event, checkbox=self.traffic_signs_checkbox: self.change_text_color(checkbox))
        
        self.car_pedestrian_var = tk.IntVar()
        self.car_pedestrian_checkbox = tk.Checkbutton(self.layout, text="Car and Pedestrian Detection", variable=self.car_pedestrian_var, command=self.save_data)
        self.car_pedestrian_checkbox.config(bg ="light blue",font=("Arial", 30), fg="black")
        self.car_pedestrian_checkbox.grid(row=5, column=0, sticky=tk.W, padx=50, pady=8)
        self.car_pedestrian_checkbox.bind("<Button-1>", lambda event, checkbox=self.car_pedestrian_checkbox: self.change_text_color(checkbox))
        
        self.disable_all_var = tk.IntVar()
        self.disable_all_checkbox = tk.Checkbutton(self.layout, text="Disable All Features", variable=self.disable_all_var, command=self.toggle_disable_all)
        self.disable_all_checkbox.config(bg ="light blue",font=("Arial", 30), fg="black")
        self.disable_all_checkbox.grid(row=6, column=0, sticky=tk.W, padx=100, pady=8)
        self.disable_all_checkbox.bind("<Button-1>", lambda event, checkbox=self.disable_all_checkbox: self.change_text_color(checkbox))

        # Add a warning label at the bottom
        self.warning_label = tk.Label(self, text="ATTENTION!\nAudio alerts will be provided based on selected features.\n*While the system is running, it may not provide %100 accurate results; errors can occur.", font=("Arial", 18, "bold"), fg="red", bg="light blue")
        self.warning_label.pack(side=tk.BOTTOM, pady=8)
        
        title_label = tk.Label(self, text="B. Aydemir, E.Potuk, G. Körpe, Y.B. Avşar",bg= "light blue", font=("Arial", 10, "bold"), fg="black")
        title_label.pack(side=tk.BOTTOM,pady=4,padx= 0 )
        



        
        # Save initial state of checkboxes
        self.initial_states = {
            'lane': self.lane_var.get(),
            'face': self.face_var.get(),
            'hand': self.hand_var.get(),
            'traffic_lights': self.traffic_lights_var.get(),
            'traffic_signs': self.traffic_signs_var.get(),
            'car_pedestrian': self.car_pedestrian_var.get()
        }

    def toggle_disable_all(self):
        # Toggle disable state of all checkboxes based on the state of the Disable All checkbox
        disable_state = not self.disable_all_var.get()
        self.lane_checkbox.config(state=tk.NORMAL if disable_state else tk.DISABLED)
        self.face_checkbox.config(state=tk.NORMAL if disable_state else tk.DISABLED)
        self.hand_checkbox.config(state=tk.NORMAL if disable_state else tk.DISABLED)
        self.traffic_lights_checkbox.config(state=tk.NORMAL if disable_state else tk.DISABLED)
        self.traffic_signs_checkbox.config(state=tk.NORMAL if disable_state else tk.DISABLED)
        self.car_pedestrian_checkbox.config(state=tk.NORMAL if disable_state else tk.DISABLED)

        if disable_state:
            # If disabling, write "0" to all files
            for key in self.initial_states:
                with open(f"{key}.txt", 'w') as file:
                    file.write("0")
        else:
            # If enabling, restore previous state and update files accordingly
            for key, value in self.initial_states.items():
                var = getattr(self, f"{key}_var")
                var.set(value)
                with open(f"{key}.txt", 'w') as file:
                    file.write("1" if value else "0")

            # Update text colors
            self.update_text_colors()

    def save_data(self):
        data = {
            'lane': "1" if self.lane_var.get() else "0",
            'face': "1" if self.face_var.get() else "0",
            'hand': "1" if self.hand_var.get() else "0",
            'traffic_lights': "1" if self.traffic_lights_var.get() else "0",
            'traffic_signs': "1" if self.traffic_signs_var.get() else "0",
            'car_pedestrian': "1" if self.car_pedestrian_var.get() else "0"
        }

        for key, value in data.items():
            with open(f"{key}.txt", 'w') as file:
                file.write(value)

    def change_text_color(self, checkbox):
        current_color = checkbox.cget("fg")
        new_color = "green" if current_color != "green" else "black"
        checkbox.config(fg=new_color)

    def update_text_colors(self):
        # Update text colors based on current state
        for key, value in self.initial_states.items():
            checkbox = getattr(self, f"{key}_checkbox")
            if value == 1:
                checkbox.config(fg="green")
            else:
                checkbox.config(fg="black")

if __name__ == "__main__":
    collector = UI()
    collector.mainloop()
