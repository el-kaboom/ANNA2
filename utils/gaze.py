# utils/gaze.py
import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh

class GazeEstimator:
    def __init__(self, max_faces=1):
        # refine_landmarks=False increases reliability on difficult frames
        self.mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=max_faces)

    def estimate(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return {"found": False, "direction": None, "score": 0.0}

        lm = res.multi_face_landmarks[0].landmark

        # landmarks for outer/inner eye corners (MediaPipe indexes)
        LEFT_EYE_OUTER = 33; LEFT_EYE_INNER = 133
        RIGHT_EYE_OUTER = 263; RIGHT_EYE_INNER = 362
        # iris landmarks may not exist when refine_landmarks=False
        # so we compute approximate eye centers
        try:
            left_outer = np.array([lm[LEFT_EYE_OUTER].x*w, lm[LEFT_EYE_OUTER].y*h])
            left_inner = np.array([lm[LEFT_EYE_INNER].x*w, lm[LEFT_EYE_INNER].y*h])
            right_outer = np.array([lm[RIGHT_EYE_OUTER].x*w, lm[RIGHT_EYE_OUTER].y*h])
            right_inner = np.array([lm[RIGHT_EYE_INNER].x*w, lm[RIGHT_EYE_INNER].y*h])

            # approximate iris centers as midpoint between inner and outer of each eye
            left_iris = (left_outer + left_inner) / 2.0
            right_iris = (right_outer + right_inner) / 2.0
        except Exception:
            return {"found": False, "direction": None, "score": 0.0}

        def gaze_ratio(iris, inner, outer):
            vec = outer - inner
            if np.linalg.norm(vec) < 1e-6:
                return 0.5
            v = np.dot(iris - inner, vec) / np.dot(vec, vec)
            return float(np.clip(v, 0.0, 1.0))

        lr = gaze_ratio(left_iris, left_inner, left_outer)
        rr = gaze_ratio(right_iris, right_inner, right_outer)
        avg = (lr + rr) / 2.0

        # simple vertical heuristic
        eye_center_y = (left_outer[1] + left_inner[1] + right_outer[1] + right_inner[1]) / 4.0
        iris_y = (left_iris[1] + right_iris[1]) / 2.0
        vertical_diff = iris_y - eye_center_y

        if avg < 0.35:
            direction = "right"; score = 0.30
        elif avg > 0.65:
            direction = "left"; score = 0.30
        else:
            direction = "center"; score = 0.95

        if vertical_diff > 10:
            direction = "down"; score = 0.25
        elif vertical_diff < -10:
            direction = "up"; score = 0.25

        return {"found": True, "direction": direction, "score": float(score)}
