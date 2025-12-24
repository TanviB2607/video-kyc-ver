import cv2
import mediapipe as mp
import numpy as np

class LivenessService:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def passive_liveness(self, frames):
        return 1.0 if len(frames) > 10 else 0.0

    def active_liveness(self, frames):
        look_left = False
        look_right = False
        blink_count = 0
        blink_state = False

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            lm = results.multi_face_landmarks[0].landmark

            nose_x = lm[1].x
            eye_mid = (lm[33].x + lm[263].x) / 2

            if nose_x < eye_mid - 0.03:
                look_left = True
            if nose_x > eye_mid + 0.03:
                look_right = True

            # Blink detection (Eye Aspect Ratio approximation)
            ear = abs(lm[159].y - lm[145].y)
            if ear < 0.015 and not blink_state:
                blink_count += 1
                blink_state = True
            elif ear > 0.02:
                blink_state = False

        return 1.0 if look_left and look_right and blink_count >= 3 else 0.0
