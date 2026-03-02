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

        self.bw = 120
        self.bh = 80
        self.offset = 50

        try:
            self.img_trash = cv2.resize(cv2.imread("TrashIcon.png", cv2.IMREAD_UNCHANGED), (self.bw, self.bh))
            self.img_erase = cv2.resize(cv2.imread("Erase.png", cv2.IMREAD_UNCHANGED), (self.bw, self.bh))
            self.img_green = cv2.resize(cv2.imread("Green.png", cv2.IMREAD_UNCHANGED), (self.bw, self.bh))
            self.img_red = cv2.resize(cv2.imread("Red.png", cv2.IMREAD_UNCHANGED), (self.bw, self.bh))
            self.icons_loaded = True
        except Exception:
            self.icons_loaded = False

    def overlay_image(self, background, overlay, x, y):
        h, w = overlay.shape[:2]

        if y + h > background.shape[0] or x + w > background.shape[1] or y < 0 or x < 0:
            return

        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            overlay_rgb = overlay[:, :, :3]
            for c in range(3):
                background[y:y + h, x:x + w, c] = (
                        alpha * overlay_rgb[:, :, c] +
                        (1 - alpha) * background[y:y + h, x:x + w, c]
                )
        else:
            background[y:y + h, x:x + w] = overlay

    def draw_line(self, frame, points, color):
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance < 100:
                cv2.line(frame, points[i - 1], points[i], color, 2)

    def draw_ui(self, frame, w, h):
        if self.icons_loaded:
            self.overlay_image(frame, self.img_trash, 0, 0)
            self.overlay_image(frame, self.img_erase, w - self.bw, 0)
            self.overlay_image(frame, self.img_green, 0, h - self.bh - self.offset)
            self.overlay_image(frame, self.img_red, w - self.bw, h - self.bh - self.offset)
        else:
            cv2.rectangle(frame, (0, 0), (self.bw, self.bh), (200, 200, 200), -1)
            cv2.rectangle(frame, (w - self.bw, 0), (w, self.bh), (200, 200, 200), -1)
            cv2.rectangle(frame, (0, h - self.bh - self.offset), (self.bw, h - self.offset), (0, 255, 0), -1)
            cv2.rectangle(frame, (w - self.bw, h - self.bh - self.offset), (w, h - self.offset), (0, 0, 255), -1)

    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        self.draw_ui(frame, w, h)

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

                if w - self.bw <= xt <= w and 0 <= yt <= self.bh:
                    if current_time - self.start_time_erase > 1:
                        self.green_color_bool = False
                        self.red_color_bool = False
                        self.erase_bool = True
                    else:
                        cv2.circle(frame, (xt, yt), int(30 * (current_time - self.start_time_erase)), (255, 255, 255),
                                   3)
                else:
                    self.start_time_erase = current_time

                if 0 <= xt <= self.bw and h - self.bh - self.offset <= yt <= h - self.offset:
                    if current_time - self.start_time_green > 1:
                        self.green_color_bool = True
                        self.red_color_bool = False
                        self.erase_bool = False
                    else:
                        cv2.circle(frame, (xt, yt), int(30 * (current_time - self.start_time_green)), (0, 255, 0), 3)
                else:
                    self.start_time_green = current_time

                if w - self.bw <= xt <= w and h - self.bh - self.offset <= yt <= h - self.offset:
                    if current_time - self.start_time_red > 1:
                        self.red_color_bool = True
                        self.green_color_bool = False
                        self.erase_bool = False
                    else:
                        cv2.circle(frame, (xt, yt), int(30 * (current_time - self.start_time_red)), (0, 0, 255), 3)
                else:
                    self.start_time_red = current_time

                if 0 <= xt <= self.bw and 0 <= yt <= self.bh:
                    if current_time - self.start_time_trash > 1:
                        self.green_points = []
                        self.red_points = []
                    else:
                        cv2.circle(frame, (xt, yt), int(30 * (current_time - self.start_time_trash)), (200, 200, 200),
                                   3)
                else:
                    self.start_time_trash = current_time

                draw_utils.draw_landmarks(frame, landmarks, hands_module.HAND_CONNECTIONS)

                drawing = yt < yp

                if self.green_color_bool and drawing:
                    self.green_points.append((xt, yt))
                elif self.red_color_bool and drawing:
                    self.red_points.append((xt, yt))
                elif self.erase_bool and drawing:
                    self.radius = 30
                    self.green_points = [
                        p for p in self.green_points if math.hypot(p[0] - xt, p[1] - yt) > self.radius
                    ]
                    self.red_points = [
                        p for p in self.red_points if math.hypot(p[0] - xt, p[1] - yt) > self.radius
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