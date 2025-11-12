# main.py
# Phase 1 of AI-Powered Eye Fatigue Detector (Improved Version)
# This script detects human faces, eyes, and smiles in real-time using a webcam feed.
# Improvements: Adjusted parameters to reduce false positives, added Gaussian blur for stability,
# added smile detection for testing, and included eye position checks.
# Requirements: OpenCV and numpy only. Uses built-in OpenCV Haar cascade files.

import cv2  # Import OpenCV for computer vision tasks
import numpy as np  # Import numpy (required, though not heavily used here)

# Load Haar Cascade classifiers for face, eye, and smile detection
# Use OpenCV's built-in data path to locate the XML files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Check if cascades loaded successfully
if face_cascade.empty():
    print("Error: Could not load face cascade. Ensure OpenCV is installed correctly.")
    exit(1)
if eye_cascade.empty():
    print("Error: Could not load eye cascade. Ensure OpenCV is installed correctly.")
    exit(1)
if smile_cascade.empty():
    print("Warning: Could not load smile cascade. Smile detection will be skipped. Ensure OpenCV is installed correctly.")

# Initialize webcam capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Check if webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam. Please check if it's connected and not in use.")
    exit(1)

print("Starting Eye Fatigue Detector - Phase 1 (Improved). Press 'q' to quit.")

# Main loop for real-time detection
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Handle frame capture errors gracefully
    if not ret:
        print("Error: Failed to capture frame from webcam. Retrying...")
        continue  # Skip to next iteration instead of crashing
    
    # Optional: Resize frame for better performance (adjust as needed)
    frame = cv2.resize(frame, (640, 480))
    
    # Convert the frame to grayscale and apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur to stabilize detection
    
    # Detect faces in the grayscale frame with adjusted parameters for stability
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(50, 50))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a blue rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color (BGR: 255,0,0)
        
        # Define the region of interest (ROI) for eye and smile detection within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect eyes within the face ROI with stricter parameters
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=15, minSize=(20, 20))
        
        # Draw green rectangles around detected eyes, but only if they are in the upper half of the face (to reduce false positives)
        for (ex, ey, ew, eh) in eyes:
            if ey < h * 0.6:  # Eyes should be in the top 60% of the face
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green color (BGR: 0,255,0)
        
        # Detect smiles within the face ROI for testing (optional feature)
        if not smile_cascade.empty():
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)  # Red color (BGR: 0,0,255)
    
    # Display the output frame in a window titled "Eye Fatigue Detector - Phase 1"
    cv2.imshow('Eye Fatigue Detector - Phase 1', frame)
    
    # Wait for a key press; exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Eye Fatigue Detector - Phase 1 exited successfully.")
