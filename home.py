
import cv2
import numpy as np
from tensorflow import keras
import tkinter as tk
from PIL import Image, ImageTk

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model
mask_model = keras.models.load_model('model.h5')

# Define labels for mask detection
labels = ["Without Mask", "With Mask"]

# Initialize tkinter window
root = tk.Tk()
root.title("Real-Time Mask Detection")
root.geometry("800x600")

# Create a label to display the camera feed
camera_label = tk.Label(root)
camera_label.pack()

# Open the default camera
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y + h, x:x + w]

        # Preprocess the face for the mask model
        face_resized = cv2.resize(face, (224, 224))  # Adjust size based on model input
        face_array = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_array = face_array / 255.0  # Normalize pixel values

        # Predict mask status
        prediction = mask_model.predict(face_array)
        mask_label = np.argmax(prediction)  # Get the index of the highest probability
        confidence = np.max(prediction)

        # Draw rectangle around face and display label
        color = (0, 255, 0) if mask_label == 1 else (0, 0, 255)  # Green for "With Mask", Red for "Without Mask"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{labels[mask_label]}: {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Convert frame to ImageTk format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    camera_label.img_tk = img_tk
    camera_label.configure(image=img_tk)

    # Repeat the function after 10 milliseconds
    root.after(10, update_frame)

# Start updating frames
update_frame()

# Run the tkinter main loop
root.mainloop()

# Release the camera when the window is closed
cap.release()
cv2.destroyAllWindows()