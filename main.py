import cv2
import mediapipe as mp

hands_module = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

hands = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1
    # min_detection_confidence=0.5,
    # min_tracking_confidence=0.5
)

points =  []


while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
             index_tip = landmarks.landmark[8]
             x = int(index_tip.x * w)
             y = int(index_tip.y * h)
             draw_utils.draw_landmarks(
                frame,
                landmarks,
                hands_module.HAND_CONNECTIONS
             )
             points.append((x, y))
             cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()