import cv2
import mediapipe as mp
import time
import math

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

green_points = []
red_points = []
green_color = (0, 255, 0)
red_color = (0, 0, 255)
red_color_bool = False
green_color_bool = True
green_last_color = True
erase_bool = False

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
        
def draw_line(points, color):
    for i in range(1, len(points)):

        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance < 100:
            cv2.line(frame, points[i - 1], points[i], color, 2)

# def erase_line(points, color):
#     for i in range(1, len(points)):




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
             index_pip = landmarks.landmark[6]

             xt = int(index_tip.x * w)
             yt = int(index_tip.y * h)
             yp = int(index_pip.y * h)

             if 1000 <= xt <= 1200 and 0 <= yt <= 100:
                 if time.perf_counter() - start_time_erase > 1:
                     green_color_bool = False
                     red_color_bool = False
                     erase_bool = True

                 else:
                     start_time_erase = time.perf_counter()

             if 0 <= xt <= 200 and 500 <= yt <= 600:
                 if time.perf_counter() - start_time_green > 1:
                     green_color_bool = True
                     green_last_color = True
                     red_color_bool = False
                     red_last_color = False

             else:
                 start_time_green = time.perf_counter()

             if 1000 <= xt <= 1200 and 500 <= yt <= 600:
                 if time.perf_counter() - start_time_red > 1:
                     red_color_bool = True
                     green_color_bool = False
                     red_last_color = True
                     green_last_color = False

             else:
                 start_time_red = time.perf_counter()

             if 0 <= xt <= 200 and 0 <= yt <= 100:
                 if time.perf_counter() - start_time_trash > 1:
                     green_points = []
                     red_points = []

             else:
                 start_time_trash = time.perf_counter()


             draw_utils.draw_landmarks(
                frame,
                landmarks,
                hands_module.HAND_CONNECTIONS
             )

             cv2.circle(frame, (xt, yt), 8, (0, 0, 255), -1)

             if yt < yp:
                 green_color_bool = False
                 red_color_bool = False

             elif green_last_color:
                 green_color_bool = True
                 red_color_bool = False

             elif red_last_color:
                 red_color_bool = True
                 green_color_bool = False

             if green_color_bool:
                 green_points.append((xt, yt))

             elif red_color_bool:
                 red_points.append((xt, yt))

    # for i in range(1, len(green_points)):
    #     cv2.line(frame, green_points[i - 1], green_points[i], green_color, 2)



    draw_line(green_points, green_color)
    draw_line(red_points, red_color)

    cv2.imshow("Virtual Drawing Bord", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()