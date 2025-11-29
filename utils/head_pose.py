# utils/head_pose.py
import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh


class HeadPoseEstimator:
    def __init__(self, max_faces=1):
        # refine_landmarks=True improves accuracy
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {"found": False, "yaw": None, "pitch": None, "roll": None, "score": 0.0}

        lm = result.multi_face_landmarks[0].landmark

        # ======== USE VERY STABLE 6 LANDMARKS ========
        # Nose tip (1)
        # Chin (152)
        # Left eye outer corner (33)
        # Right eye outer corner (263)
        # Left mouth corner (61)
        # Right mouth corner (291)
        try:
            image_points = np.array([
                (lm[1].x * w, lm[1].y * h),  # nose tip
                (lm[152].x * w, lm[152].y * h),  # chin
                (lm[33].x * w, lm[33].y * h),  # left eye outer
                (lm[263].x * w, lm[263].y * h),  # right eye outer
                (lm[61].x * w, lm[61].y * h),  # left mouth corner
                (lm[291].x * w, lm[291].y * h)  # right mouth corner
            ], dtype="double")
        except Exception:
            return {"found": False, "yaw": None, "pitch": None, "roll": None, "score": 0.0}

        # 3D head model (generic human face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -63.0, 0.0),  # Chin
            (-34.0, 32.0, -30.0),  # Left eye outer
            (34.0, 32.0, -30.0),  # Right eye outer
            (-40.0, -28.0, -30.0),  # Left mouth corner
            (40.0, -28.0, -30.0)  # Right mouth corner
        ])

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        try:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return {"found": False, "yaw": None, "pitch": None, "roll": None, "score": 0.0}

            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])

            pitch_deg = abs(np.degrees(x))
            yaw_deg = abs(np.degrees(y))

            # Very stable scoring
            yaw_score = max(0.0, 1.0 - yaw_deg / 40.0)
            pitch_score = max(0.0, 1.0 - pitch_deg / 40.0)

            score = (yaw_score + pitch_score) / 2.0

            return {
                "found": True,
                "yaw": float(y),
                "pitch": float(x),
                "roll": float(z),
                "score": float(score)
            }

        except Exception:
            return {"found": False, "yaw": None, "pitch": None, "roll": None, "score": 0.0}
