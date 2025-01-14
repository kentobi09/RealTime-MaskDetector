import cv2
import numpy as np
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from datetime import datetime
from collections import deque

class MaskDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Real-Time Mask Detection System - Team LGTV")
        self.window.state('zoomed')  # Full screen
        
        # Load models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mask_model = keras.models.load_model('model.h5')
        self.labels = ["Without Mask", "With Mask"]
        
        # Initialize counters
        self.mask_count = 0
        self.no_mask_count = 0
        self.last_alert_time = 0
        
        # Detection state tracking
        self.face_states = {}  # Dictionary to track face detection states
        self.detection_threshold = 3.0  # Seconds needed for consistent detection
        self.cleanup_threshold = 5.0  # Seconds after which to remove stale face tracks
        
        self.setup_ui()
        self.initialize_camera()
        
    def setup_ui(self):
        # UI setup code remains the same as before
        self.top_frame = ttk.Frame(self.window)
        self.top_frame.pack(fill='x', padx=10, pady=5)
        
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Title
        title_label = ttk.Label(self.top_frame, 
                               text="Real-Time Face Mask Detector", 
                               font=('Helvetica', 24, 'bold'))
        title_label.pack(pady=10)
        
        # Camera feed frame
        self.camera_frame = ttk.Frame(self.main_frame)
        self.camera_frame.pack(side='left', expand=True, fill='both')
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Stats frame
        stats_frame = ttk.Frame(self.main_frame)
        stats_frame.pack(side='right', fill='y', padx=20, pady=10)
        
        # Statistics
        ttk.Label(stats_frame, text="Statistics", 
                 font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        self.mask_count_label = ttk.Label(stats_frame, 
                                        text="People with Mask: 0",
                                        font=('Helvetica', 12))
        self.mask_count_label.pack(pady=5)
        
        self.no_mask_count_label = ttk.Label(stats_frame,
                                           text="People without Mask: 0",
                                           font=('Helvetica', 12))
        self.no_mask_count_label.pack(pady=5)
        
        # Status frame
        self.status_frame = ttk.Frame(stats_frame)
        self.status_frame.pack(pady=20)
        
        self.status_label = ttk.Label(self.status_frame,
                                    text="System Status: Monitoring",
                                    font=('Helvetica', 12))
        self.status_label.pack()
        
        # Alert frame
        self.alert_frame = ttk.Frame(stats_frame, relief='ridge', borderwidth=2)
        self.alert_frame.pack(pady=20, fill='x')
        
        self.alert_label = ttk.Label(self.alert_frame,
                                   text="No Alerts",
                                   font=('Helvetica', 12))
        self.alert_label.pack(pady=10)

    def get_face_id(self, x, y, w, h):
        """Generate a unique ID for a face based on its position"""
        center_x = x + w/2
        center_y = y + h/2
        return f"{int(center_x)}_{int(center_y)}"
    
    def cleanup_stale_faces(self, current_time):
        """Remove face tracks that haven't been seen recently"""
        stale_faces = []
        for face_id, state in self.face_states.items():
            if current_time - state['last_seen'] > self.cleanup_threshold:
                stale_faces.append(face_id)
        
        for face_id in stale_faces:
            del self.face_states[face_id]
            
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="System Status: Camera Error")
            return False
        self.status_label.config(text="System Status: Camera Active")
        return True
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="System Status: Frame Capture Error")
            return
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 
                                                 scaleFactor=1.1, 
                                                 minNeighbors=5, 
                                                 minSize=(30, 30))
        
        current_time = time.time()
        
        # Cleanup old face tracks
        self.cleanup_stale_faces(current_time)
        
        # Track current faces
        current_faces = set()
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Initialize color as orange (analyzing state)
            color = (255, 165, 0)
            label = "Analyzing..."
            
            face_id = self.get_face_id(x, y, w, h)
            current_faces.add(face_id)
            
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            face_array = np.expand_dims(face_resized, axis=0) / 255.0
            
            # Predict mask status
            prediction = self.mask_model.predict(face_array, verbose=0)
            mask_label = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Initialize or update face state
            if face_id not in self.face_states:
                self.face_states[face_id] = {
                    'mask_history': deque(maxlen=30),  # Store last 30 detections
                    'start_time': current_time,
                    'last_seen': current_time,
                    'counted': False
                }
            else:
                self.face_states[face_id]['last_seen'] = current_time
            
            # Update detection history
            self.face_states[face_id]['mask_history'].append(mask_label)
            
            # Calculate detection duration
            detection_duration = current_time - self.face_states[face_id]['start_time']
            
            # Determine predominant mask status over the last 3 seconds
            if detection_duration >= self.detection_threshold and not self.face_states[face_id]['counted']:
                mask_history = self.face_states[face_id]['mask_history']
                if len(mask_history) >= 15:  # Ensure we have enough samples
                    mask_ratio = sum(1 for x in mask_history if x == 1) / len(mask_history)
                    
                    if mask_ratio > 0.7:  # More than 70% detections show mask
                        self.mask_count += 1
                        color = (0, 255, 0)
                        self.face_states[face_id]['counted'] = True
                        label = f"{self.labels[1]}: {confidence:.2f}"
                    elif mask_ratio < 0.3:  # More than 70% detections show no mask
                        self.no_mask_count += 1
                        color = (0, 0, 255)
                        self.face_states[face_id]['counted'] = True
                        label = f"{self.labels[0]}: {confidence:.2f}"
                        
                        # Trigger alert
                        if current_time - self.last_alert_time >= 3:
                            self.alert_label.config(text=f"⚠️ Alert: Person detected without mask at {datetime.now().strftime('%H:%M:%S')}")
                            self.last_alert_time = current_time
            
            # If still analyzing, show countdown
            if not self.face_states[face_id]['counted']:
                remaining_time = max(0, self.detection_threshold - detection_duration)
                label = f"Analyzing: {remaining_time:.1f}s"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update statistics
        self.mask_count_label.config(text=f"People with Mask: {self.mask_count}")
        self.no_mask_count_label.config(text=f"People without Mask: {self.no_mask_count}")
        
        # Convert frame for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.camera_label.img_tk = img_tk
        self.camera_label.config(image=img_tk)
        
        # Schedule next update
        self.window.after(10, self.update_frame)
    
    def run(self):
        if self.initialize_camera():
            self.update_frame()
        self.window.mainloop()
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionApp(root)
    app.run()