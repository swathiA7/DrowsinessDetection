import cv2
import winsound  # For playing alert sound

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Constants for drowsiness detection
EYE_CLOSED_FRAMES = 20  # Frames threshold for drowsiness alert
eye_closed_counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    drowsy_state = "NOT DROWSY"  # Default state

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        # Check if eyes are detected
        if len(eyes) == 0:
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0  # Reset counter if eyes are open

        # Determine drowsy state
        if eye_closed_counter >= EYE_CLOSED_FRAMES:
            drowsy_state = "DROWSY"
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Play an alert sound
            winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

    # Display the drowsy state on the screen
    cv2.putText(frame, f"State: {drowsy_state}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if drowsy_state == "NOT DROWSY" else (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
