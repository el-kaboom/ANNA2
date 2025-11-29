# utils/activity.py
from ultralytics import YOLO
import numpy as np
import cv2

class ActivityDetector:
    def __init__(self, model_name="yolov8n.pt"):
        # auto-downloads yolov8n.pt on first run
        self.model = YOLO(model_name)
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame, head_pose_result=None, face_found=True):
        """
        Returns {using_phone, sleeping, no_face, details}
        """
        # default
        using_phone = False
        phone_boxes = []

        try:
            results = self.model.predict(frame, imgsz=640, conf=0.35, verbose=False)
            for r in results:
                boxes = getattr(r, "boxes", [])
                for box in boxes:
                    cls = int(box.cls[0])
                    name = self.names.get(cls, str(cls))
                    if "phone" in name.lower() or "cell phone" in name.lower() or "cellphone" in name.lower():
                        using_phone = True
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        phone_boxes.append({"name": name, "xyxy": xyxy})
        except Exception:
            # YOLO can fail on some systems temporarily; don't crash
            pass

        sleeping = False
        if head_pose_result and head_pose_result.get("found"):
            pitch = head_pose_result.get("pitch", 0.0)
            pitch_deg = abs(pitch * 180.0 / 3.14159265)
            if pitch_deg > 45:
                sleeping = True

        return {"using_phone": using_phone, "sleeping": sleeping, "no_face": not face_found, "details": {"phones": phone_boxes}}
