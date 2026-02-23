from imaplib import Debug
import cv2
import mediapipe as mp
import time

hands_module = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

TrashCan = cv2.imread("TrashIcon.png", cv2.IMREAD_UNCHANGED)
TrashCan = cv2.resize(TrashCan, (200, 100))

Erase = cv2.imread("Erase.png", cv2.IMREAD_UNCHANGED)
Erase = cv2.resize(Erase, (200, 100))

Green = cv2.imread("Green.png", cv2.IMREAD_UNCHANGED)
Green = cv2.resize(Green, (200, 100))

Red = cv2.imread("Red.png", cv2.IMREAD_UNCHANGED)
Red = cv2.resize(Red, (200, 100))

hands = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1
    # min_detection_confidence=0.5,
    # min_tracking_confidence=0.5
)

def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        overlay_rgb = overlay[:, :, :3]

        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay_rgb[:, :, c] +
                (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay

points =  []




while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    overlay_image(frame, TrashCan, 0, 0)
    overlay_image(frame, Erase, 1000, 0)
    overlay_image(frame, Green, 0,500)
    overlay_image(frame, Red, 1000, 500)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
             index_tip = landmarks.landmark[8]
             x = int(index_tip.x * w)
             y = int(index_tip.y * h)

             if 1000 <= x <= 1200 and 0 <= y <= 100:
                 print("Hand is over Erase!")

             if 0 <= x <= 200 and 500 <= y <= 600:
                 print("Color changed to Green!")

             if 1000 <= x <= 1200 and 500 <= y <= 600:
                 print("Color changed to Red!")

             if 0 <= x <= 200 and 0 <= y <= 100:
                 if time.perf_counter() - start_time > 1:
                     points = []

             else:
                 start_time = time.perf_counter()


             draw_utils.draw_landmarks(
                frame,
                landmarks,
                hands_module.HAND_CONNECTIONS
             )
             points.append((x, y))
             cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

    cv2.imshow("Virtual Drawing Bord", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()