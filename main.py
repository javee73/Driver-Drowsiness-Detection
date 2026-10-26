
import cv2
import mediapipe as mp
import math
import os
from pygame import mixer

# ==========================
# ALARM SETUP
# ==========================

ALARM_FILE = "alarm.wav"
ALARM_ON = False

try:
    mixer.init()
    print("Audio system ready!")
except Exception as e:
    print("Audio initialization error:", e)

# ==========================
# MEDIAPIPE FACE MESH
# ==========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# EYE AND MOUTH LANDMARKS
# ==========================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14

# ==========================
# THRESHOLDS
# ==========================

EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 20

YAWN_THRESHOLD = 25

COUNTER = 0

# ==========================
# FUNCTIONS
# ==========================

def euclidean(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)


# ==========================
# CHECK ALARM FILE
# ==========================

if not os.path.exists(ALARM_FILE):
    print("WARNING: alarm.wav file not found!")
    print("Place alarm.wav in the same folder as main.py.")
    ALARM_AVAILABLE = False
else:
    ALARM_AVAILABLE = True

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    face_mesh.close()
    if mixer.get_init():
        mixer.quit()
    raise SystemExit

print("Camera started. Press Q to exit.")

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                points = []

                for lm in face_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))

                # ======================
                # EYE DETECTION
                # ======================

                left_eye = [points[i] for i in LEFT_EYE]
                right_eye = [points[i] for i in RIGHT_EYE]

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)

                ear = (leftEAR + rightEAR) / 2.0

                for p in left_eye + right_eye:
                    cv2.circle(frame, p, 2, (0, 255, 0), -1)

                # ======================
                # DROWSINESS CHECK
                # ======================

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                else:
                    COUNTER = 0

                    if ALARM_ON:
                        if mixer.get_init():
                            mixer.music.stop()
                        ALARM_ON = False
                        print("Alarm stopped.")

                if COUNTER >= EAR_CONSEC_FRAMES:

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    if not ALARM_ON and ALARM_AVAILABLE and mixer.get_init():
                        try:
                            mixer.music.load(ALARM_FILE)
                            mixer.music.play(-1)
                            ALARM_ON = True
                            print("DROWSINESS ALERT! Alarm started.")
                        except Exception as e:
                            print("Alarm playback error:", e)

                else:
                    cv2.putText(
                        frame,
                        "Eyes: OPEN",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                # ======================
                # YAWN DETECTION
                # ======================

                upper_lip = points[UPPER_LIP]
                lower_lip = points[LOWER_LIP]

                mouth_open = abs(upper_lip[1] - lower_lip[1])

                cv2.line(
                    frame,
                    upper_lip,
                    lower_lip,
                    (255, 0, 0),
                    2
                )

                if mouth_open > YAWN_THRESHOLD:
                    cv2.putText(
                        frame,
                        "MOUTH OPEN",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )

                # ======================
                # INFO DISPLAY
                # ======================

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Counter: {COUNTER}/{EAR_CONSEC_FRAMES}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        else:
            COUNTER = 0

            if ALARM_ON:
                if mixer.get_init():
                    mixer.music.stop()
                ALARM_ON = False

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Driver Drowsiness Detection System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if ALARM_ON and mixer.get_init():
        mixer.music.stop()

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if mixer.get_init():
        mixer.quit()

    print("Project closed safely.")
