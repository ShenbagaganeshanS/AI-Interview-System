import cv2
from deepface import DeepFace

def detect_emotion_from_frame(frame):
    """Detect emotion from a single frame. Returns emotion string."""
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        return result[0]['dominant_emotion']
    except Exception as e:
        print("Emotion detection error:", e)
        return "unknown"


def run_live_camera():
    """Run live webcam emotion detection (standalone use)."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion_from_frame(frame)
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Interview Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_camera()