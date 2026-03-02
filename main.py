import streamlit as st
import cv2
import mediapipe as mp
import av
import math
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.title("🖐️ Virtual Drawing Board")

hands_module = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils


class DrawingProcessor(VideoProcessorBase):

    def __init__(self):

        self.hands = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1
        )

        self.green_points = []
        self.red_points = []

        self.green_color = (0, 255, 0)
        self.red_color = (0, 0, 255)

        self.red_color_bool = False
        self.green_color_bool = True
        self.erase_bool = False

        self.radius = 8

        self.start_time_erase = 0
        self.start_time_green = 0
        self.start_time_red = 0
        self.start_time_trash = 0

    def draw_line(self, frame, points, color):
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance < 100:
                cv2.line(frame, points[i - 1], points[i], color, 2)

    def draw_ui(self, frame):

        # Trash
        cv2.rectangle(frame, (0, 0), (200, 100), (200, 200, 200), -1)
        cv2.putText(frame, "TRASH", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Erase
        cv2.rectangle(frame, (1000, 0), (1200, 100), (200, 200, 200), -1)
        cv2.putText(frame, "ERASE", (1020, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Green
        cv2.rectangle(frame, (0, 500), (200, 600), (0, 255, 0), -1)
        cv2.putText(frame, "GREEN", (20, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Red
        cv2.rectangle(frame, (1000, 500), (1200, 600), (0, 0, 255), -1)
        cv2.putText(frame, "RED", (1050, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def recv(self, frame):

        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        self.draw_ui(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for landmarks in results.multi_hand_landmarks:

                index_tip = landmarks.landmark[8]
                index_pip = landmarks.landmark[6]

                xt = int(index_tip.x * w)
                yt = int(index_tip.y * h)
                yp = int(index_pip.y * h)

                current_time = time.perf_counter()

                # ERASE
                if 1000 <= xt <= 1200 and 0 <= yt <= 100:
                    if current_time - self.start_time_erase > 1:
                        self.green_color_bool = False
                        self.red_color_bool = False
                        self.erase_bool = True
                    else:
                        self.start_time_erase = current_time
                else:
                    self.start_time_erase = current_time

                # GREEN
                if 0 <= xt <= 200 and 500 <= yt <= 600:
                    if current_time - self.start_time_green > 1:
                        self.green_color_bool = True
                        self.red_color_bool = False
                        self.erase_bool = False
                    else:
                        self.start_time_green = current_time
                else:
                    self.start_time_green = current_time

                # RED
                if 1000 <= xt <= 1200 and 500 <= yt <= 600:
                    if current_time - self.start_time_red > 1:
                        self.red_color_bool = True
                        self.green_color_bool = False
                        self.erase_bool = False
                    else:
                        self.start_time_red = current_time
                else:
                    self.start_time_red = current_time

                # TRASH
                if 0 <= xt <= 200 and 0 <= yt <= 100:
                    if current_time - self.start_time_trash > 1:
                        self.green_points = []
                        self.red_points = []
                    else:
                        self.start_time_trash = current_time
                else:
                    self.start_time_trash = current_time

                draw_utils.draw_landmarks(
                    frame,
                    landmarks,
                    hands_module.HAND_CONNECTIONS
                )

                if yt > yp:
                    drawing = False
                else:
                    drawing = True

                if self.green_color_bool and drawing:
                    self.green_points.append((xt, yt))

                elif self.red_color_bool and drawing:
                    self.red_points.append((xt, yt))

                elif self.erase_bool and drawing:

                    self.radius = 30

                    self.green_points = [
                        point for point in self.green_points
                        if math.hypot(point[0] - xt, point[1] - yt) > self.radius
                    ]

                    self.red_points = [
                        point for point in self.red_points
                        if math.hypot(point[0] - xt, point[1] - yt) > self.radius
                    ]

                cv2.circle(frame, (xt, yt), 8, (0, 0, 255), -1)

        self.draw_line(frame, self.green_points, self.green_color)
        self.draw_line(frame, self.red_points, self.red_color)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="virtual-drawing-board",
    video_processor_factory=DrawingProcessor,
    rtc_configuration=RTC_CONFIGURATION,
)