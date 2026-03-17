# 🖐️ Virtual Drawing Board

A real-time hand-tracking virtual drawing board built with Streamlit, MediaPipe, and OpenCV.

Draw in the air using your index finger and switch between tools using gesture-based interaction.

---
## Features
-  Draw using index finger tracking
-  Switch between Green and Red colors
-  Erase drawn points
-  Clear the entire board
-  Custom PNG UI icons with transparency
-  Runs in browser using WebRTC
-  Real-time webcam processing
-  It only tracks one hand
-  If you close your hand, it stops drawing 

## Built With
- Streamlit
- streamlit-webrtc
- OpenCV
- MediaPipe
- PyAV

---
## Demo

    https://drive.google.com/file/d/1VTF_JS8J0XWktsLya8Govyjcw2cDktdH/view

---

## Installation (Local and using Streamlit)

1. Clone repository:

   ```bash
   git clone https://github.com/flavius841/VirtualDrawingBord.git
   cd VirtualDrawingBord
   ```
   
2. Create virtual environment:
  
   ```bash 
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows
   ```

   if you can't create a virtual environment on Windows run this:

   ```bash 
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   
4. Install dependencies:
 
   ```bash 
   pip install -r requirements.txt
   ```
   
5. Run the app:
  
   ```bash 
   streamlit run main.py
   ```
   
   open the link given

   make sure you have installed streamlit:

   ```bash 
   pip install streamlit
   ```

---
 
## Project Evolution

This project was originally developed as a **local desktop application** using OpenCV and MediaPipe.

The first version used `cv2.VideoCapture()` to access the webcam directly and rendered the interface using `cv2.imshow()`. It was designed to run locally on a machine with GUI support.

After validating the core hand-tracking and drawing logic, the project was refactored and migrated to a web-based architecture using Streamlit and WebRTC. This allowed the application to run entirely in the browser, making it easier to share and deploy online.

The original desktop implementation is shown below:

```python
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
)

green_points = []
red_points = []
green_color = (0, 255, 0)
red_color = (0, 0, 255)
red_color_bool = False
green_color_bool = True
erase_bool = False
radius = 8

start_time_erase = 0
start_time_green = 0
start_time_red = 0
start_time_trash = 0

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
                     red_color_bool = False
                     erase_bool = False

             else:
                 start_time_green = time.perf_counter()

             if 1000 <= xt <= 1200 and 500 <= yt <= 600:
                 if time.perf_counter() - start_time_red > 1:
                     red_color_bool = True
                     green_color_bool = False
                     erase_bool = False

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

             cv2.circle(frame, (xt, yt), radius, (0, 0, 255), -1)
             radius = 8

             if yt > yp:
                 drawing = False

             else:
                 drawing = True

             if green_color_bool and drawing:
                 green_points.append((xt, yt))

             elif red_color_bool and drawing:
                 red_points.append((xt, yt))


             elif erase_bool and drawing:

                 radius = 30

                 green_points = [

                     point for point in green_points

                     if math.hypot(point[0] - xt, point[1] - yt) > radius

                 ]

                 red_points = [

                     point for point in red_points

                     if math.hypot(point[0] - xt, point[1] - yt) > radius

                 ]

    draw_line(green_points, green_color)
    draw_line(red_points, red_color)

    cv2.imshow("Virtual Drawing Bord", frame)

    if cv2.waitKey(1) == 27:
        break

    if cv2.getWindowProperty("Virtual Drawing Bord", cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
```
Also if you want to run it you need to make sure that the following image files are located in the **project root directory** (same folder as the source code):

-TrashIcon.png

-Erase.png

-Green.png

-Red.png

> **Note:** The original desktop OpenCV implementation may provide smoother visualization and slightly better graphical rendering quality compared to the Streamlit web-based version. However, the web version was developed to improve accessibility and deployment convenience.

---
## How It Works

- MediaPipe detects hand landmarks in real time
- Index finger tip position is tracked
- If finger tip is above PIP joint → drawing mode
- Hovering over UI buttons for 1 second activates tool
- Points are stored and connected using OpenCV lines

---

## Controls 
| Action | Gesture |
|--------|---------|
| Draw | Raise index finger |
| Switch to Green | Hover over green button (1s) |
| Switch to Red | Hover over red button (1s) |
| Erase | Hover over erase button (1s) |
| Clear All | Hover over trash button (1s) |

## Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Select `main.py`
5. Deploy
