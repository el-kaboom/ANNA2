# utils/score.py
class EMA:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

def compute_score(gaze_score, head_score, emotion_score, activity_flags):
    # all inputs expected 0..1
    w_gaze = 0.35
    w_head = 0.25
    w_emo  = 0.25
    w_act  = 0.15

    base = w_gaze * gaze_score + w_head * head_score + w_emo * emotion_score

    # gentler penalties
    if activity_flags.get("no_face", False):
        penalty = 0.7
    elif activity_flags.get("sleeping", False):
        penalty = 0.5
    elif activity_flags.get("using_phone", False):
        penalty = 0.4
    else:
        penalty = 0.0

    act_component = w_act * (0.0 if penalty > 0 else 1.0)
    raw = base + act_component
    if penalty > 0:
        raw = raw * (1.0 - penalty)

    # clamp and convert to 0..100
    final = max(0.0, min(1.0, raw))
    return round(final * 100.0, 2)
