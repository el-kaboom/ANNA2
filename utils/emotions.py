# utils/emotions.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class EmotionDetector:
    """
    Loads local Keras model placed at utils/expression_model_final.keras
    Preprocesses inputs exactly to (1,48,48,1) as required by your model.
    """

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "expression_model_final.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Emotion model not found at {model_path}")
        self.model = load_model(model_path)
        # Confirm model expects 48x48x1
        expected = getattr(self.model, "input_shape", None)
        # optional: warn if model shape differs
        if expected and not (expected[1] == 48 and expected[2] == 48 and expected[3] == 1):
            # not fatal — but warn
            print(f"[Warning] model.input_shape = {expected}, code will still try 48x48x1 preprocessing.")

        # labels - adjust if your model uses different ordering
        self.labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

    def _preprocess_for_48(self, img):
        """
        Ensure img -> grayscale 48x48x1 float32 normalized
        Accepts color BGR images or small face crops.
        """
        if img is None or img.size == 0:
            return None
        # if input is color, convert to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # sometimes face box is tiny; reject very small
        h, w = gray.shape[:2]
        if h < 10 or w < 10:
            return None

        # resize to 48x48 (use INTER_AREA for shrinking)
        try:
            r = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

        r = r.astype("float32") / 255.0
        r = np.expand_dims(r, axis=-1)   # (48,48,1)
        r = np.expand_dims(r, axis=0)    # (1,48,48,1)
        return r

    def detect(self, frame):
        """
        Input: BGR image (preferably face crop). Returns:
          {"found": bool, "emotion": str or None, "score": float}
        """
        if frame is None or frame.size == 0:
            return {"found": False, "emotion": None, "score": 0.0}

        x = self._preprocess_for_48(frame)
        if x is None:
            return {"found": False, "emotion": None, "score": 0.0}

        # Safe predict: ensure dtype float32
        x = x.astype("float32")
        try:
            preds = self.model.predict(x, verbose=0)  # returns (1,7) typically
            if preds is None or len(preds) == 0:
                return {"found": False, "emotion": None, "score": 0.0}
            preds = preds[0]
            idx = int(np.argmax(preds))
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            conf = float(preds[idx])
            # map to engagement-friendly score
            if label.lower() in ["happy", "neutral", "surprise"]:
                score = conf
            else:
                score = conf * 0.45
            return {"found": True, "emotion": label, "score": float(score)}
        except Exception as e:
            # don't crash on TF exceptions — return not-found
            print(f"[EmotionDetector] prediction error: {e}")
            return {"found": False, "emotion": None, "score": 0.0}
