# app.py
import cv2
import time
from utils.gaze import GazeEstimator
from utils.head_pose import HeadPoseEstimator
from utils.emotions import EmotionDetector
from utils.activity import ActivityDetector
from utils.score import compute_score, EMA
import mediapipe as mp

def main():
    gaze = GazeEstimator()
    head = HeadPoseEstimator()
    emo = EmotionDetector()   # will load utils/expression_model_final.keras
    act = ActivityDetector()  # YOLOv8n

    mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    gaze_ema = EMA(alpha=0.6)
    head_ema = EMA(alpha=0.6)
    emo_ema = EMA(alpha=0.6)
    eng_ema = EMA(alpha=0.6)

    missed_face_count = 0
    MISS_THRESHOLD = 6

    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # optional resize for faster processing
        if max(w, h) > 1280:
            scale = 1280.0 / max(w, h)
            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = frame.shape[:2]

        # Primary detectors (use whole frame — internals use landmarks)
        head_res = head.estimate(frame)
        gaze_res = gaze.estimate(frame)

        face_found = bool(head_res.get("found")) or bool(gaze_res.get("found"))

        face_crop = None
        x1 = y1 = x2 = y2 = None

        # If face not found by face-mesh, fallback to FaceDetection for bbox
        if not face_found:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fd_res = mp_face_detection.process(rgb)
            if fd_res.detections:
                det = fd_res.detections[0]
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                x2 = x1 + bw
                y2 = y1 + bh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 - x1 > 10 and y2 - y1 > 10:
                    face_crop = frame[y1:y2, x1:x2].copy()
                    face_found = True
                    missed_face_count = 0
            else:
                missed_face_count += 1
        else:
            missed_face_count = 0
            # optional: derive face_crop from face mesh (if you implemented coordinates)
            # We'll not rely on it here; emotion detector can accept full frame fallback if needed.

        # Emotion detection: prefer cropped face
        if face_crop is not None:
            emo_res = emo.detect(face_crop)
        else:
            # try detect on center crop (safer than whole image)
            # compute central square region
            cw = int(min(w, h) * 0.6)
            cx1 = max(0, w//2 - cw//2)
            cy1 = max(0, h//2 - cw//2)
            cx2 = cx1 + cw
            cy2 = cy1 + cw
            center_crop = frame[cy1:cy2, cx1:cx2]
            emo_res = emo.detect(center_crop)

            if not emo_res.get("found"):
                emo_res = {"found": False, "emotion": None, "score": 0.0}

        # Activity detection (phone + sleeping heuristic)
        act_res = act.detect(frame, head_pose_result=head_res, face_found=face_found)
        if missed_face_count >= MISS_THRESHOLD:
            act_res["no_face"] = True
        else:
            act_res["no_face"] = False

        # Scores (use detector-provided scores)
        gaze_score = gaze_res.get("score", 0.0) if gaze_res.get("found") else 0.0
        head_score = head_res.get("score", 0.0) if head_res.get("found") else 0.0
        emo_score = emo_res.get("score", 0.0) if emo_res.get("found") else 0.0

        # apply EMA smoothing
        gaze_s = gaze_ema.update(gaze_score)
        head_s = head_ema.update(head_score)
        emo_s = emo_ema.update(emo_score)

        engagement = compute_score(gaze_s, head_s, emo_s, act_res)
        engagement = eng_ema.update(engagement)

        # draw face bbox fallback if available
        if x1 is not None and y1 is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # overlays
        y0 = 24
        dy = 28
        cv2.putText(frame, f"Engagement: {engagement:.2f}%", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160,255,255), 2)
        cv2.putText(frame, f"Gaze: {gaze_res.get('direction')} ({gaze_s:.2f})", (10, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"HeadScore: {head_s:.2f}", (10, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Emotion: {emo_res.get('emotion')} ({emo_s:.2f})", (10, y0+3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        act_label = "OK"
        if act_res.get("no_face"):
            act_label = "No face"
        elif act_res.get("sleeping"):
            act_label = "Sleeping/HeadDown"
        elif act_res.get("using_phone"):
            act_label = "Using Phone"
        cv2.putText(frame, f"Activity: {act_label}", (10, y0+4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

        # fps
        now = time.time()
        fps = 1.0 / (now - fps_time) if now - fps_time > 0 else 0.0
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y0+5*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow("Engagement Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
