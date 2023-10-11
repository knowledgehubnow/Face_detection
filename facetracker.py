import cv2
import dlib
from math import hypot

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to display advertisement
def display_advertisement(frame, message):
    cv2.putText(frame, message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

# Function to display advertisement for smile
def toggle_smile(frame):
    display_advertisement(frame, 'Smiling')

# Function to display advertisement for blink
def toggle_blink(frame):
    display_advertisement(frame, 'Blinking')

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    ear = (A + B) / (2.0 * C)
    return ear

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Extract coordinates of eyes and mouth
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        upper_lip = landmarks.part(51).x, landmarks.part(51).y
        lower_lip = landmarks.part(57).x, landmarks.part(57).y

        # Calculate the eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Draw rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw rectangles around eyes
        for eye_landmarks in [range(36, 42), range(42, 48)]:
            eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_landmarks]
            left, top = min(eye_points, key=lambda p: p[0])
            right, bottom = max(eye_points, key=lambda p: p[0])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw rectangles around smiles
        for (sx, sy, sw, sh) in [(51, 50, 30, 10), (48, 50, 30, 10)]:
            cv2.rectangle(frame, (landmarks.part(sx).x, landmarks.part(sx).y),
                          (landmarks.part(sx).x + sw, landmarks.part(sx).y + sh), (0, 255, 0), 2)
            cv2.rectangle(frame, (landmarks.part(sy).x, landmarks.part(sy).y),
                          (landmarks.part(sy).x + sw, landmarks.part(sy).y + sh), (0, 255, 0), 2)

        # Check for blinking
        if left_ear < 0.2 and right_ear < 0.2:
            toggle_blink(frame)
        else:
            # Don't show blink toggle if not blinking
            cv2.putText(frame, '', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Check for smile (you may need to adjust the threshold)
        lip_distance = lower_lip[1] - upper_lip[1]
        if lip_distance > 20:
            toggle_smile(frame)
        else:
            # Don't show smile toggle if not smiling
            cv2.putText(frame, '', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
