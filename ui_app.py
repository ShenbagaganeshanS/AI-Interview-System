import streamlit as st
import cv2
import threading
import time
import av
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from speech_to_text import get_speech_input
from nlp import evaluate_answer
from voice import analyze_voice

st.set_page_config(page_title="AI Interview Evaluator", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0f14 0%, #1a1a2e 100%); color: #e8e8f0; }
    h1, h2, h3 { color: #a78bfa !important; }
    .metric-card {
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 10px;
    }
    .metric-label { font-size:0.78rem; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
    .metric-value { font-size:1.4rem; font-weight:700; color:#a78bfa; }
    .score-ring  { font-size:3rem; font-weight:700; }
    .question-box {
        background: rgba(124,58,237,0.1); border-left: 4px solid #7c3aed;
        border-radius: 0 10px 10px 0; padding: 16px 20px;
        font-size: 1.05rem; font-weight: 600; color: #c4b5fd; margin-bottom: 16px;
    }
    .emotion-live {
        font-size: 1.05rem; font-weight: 600; color: #a78bfa;
        background: rgba(167,139,250,0.1); border-radius: 8px;
        padding: 6px 14px; margin-top: 8px; display:inline-block;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a21caf);
        color: white; border: none; border-radius: 10px;
        padding: 12px 28px; font-size: 1.05rem; font-weight: 600;
        font-family: 'Space Grotesk', sans-serif; width: 100%;
    }
    /* Hide the default webrtc footer text */
    .css-1v0mbdj > p { display: none; }
</style>
""", unsafe_allow_html=True)

QUESTIONS = [
    {
        "question": "What is Machine Learning?",
        "reference": "Machine learning is a method where models learn from data to make predictions or decisions."
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "reference": "Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data."
    },
    {
        "question": "Explain a recent project you have worked on.",
        "reference": "I worked on a project involving data analysis, problem solving, collaboration, and delivering results."
    }
]

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class EmotionProcessor(VideoProcessorBase):
    """
    Processes every video frame from the WebRTC stream.
    Detects emotion every N frames and overlays it on the video.
    Stores latest emotion and history for the main thread to read.
    """
    def __init__(self):
        self._lock         = threading.Lock()
        self.latest_emotion = "detecting..."
        self.emotion_history = []
        self._frame_count   = 0
        self._collecting    = False   

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        if self._frame_count % 15 == 0:
            try:
                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True
                )
                emotion = result[0]["dominant_emotion"]
            except Exception:
                emotion = self.latest_emotion

            with self._lock:
                self.latest_emotion = emotion
                if self._collecting:
                    self.emotion_history.append(emotion)

        with self._lock:
            label = self.latest_emotion.upper()

        cv2.rectangle(img, (0, 0), (300, 40), (20, 10, 40), -1)
        cv2.putText(img, f"Emotion: {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 130, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def start_collecting(self):
        with self._lock:
            self.emotion_history = []
            self._collecting = True

    def stop_collecting(self):
        with self._lock:
            self._collecting = False

    def get_history(self):
        with self._lock:
            return list(self.emotion_history)

    def get_emotion(self):
        with self._lock:
            return self.latest_emotion


for k, v in [
    ("q_index",        0),
    ("results",        []),
    ("stage",          "idle"),    # idle | recording | show_result | done
    ("pending_result", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


def _speech_worker(bucket):
    
    bucket["answer"] = get_speech_input(timeout=30, phrase_limit=60)

def _voice_worker(bucket):
    bucket["voice"] = analyze_voice(duration=6)   # 6 sec for better pitch/energy sample


def dominant_emotion(history):
    return max(set(history), key=history.count) if history else "neutral"

def calc_score(a, v, e):
    s  = 40 if a == "GOOD"    else 25 if a == "AVERAGE" else 10
    s += 30 if "HIGH" in v    else 20 if "MEDIUM" in v  else 10
    s += 30 if e in ["happy", "neutral"] else 18 if e == "surprise" else 10
    return s

def score_color(s):
    return "#4ade80" if s > 75 else "#fbbf24" if s > 50 else "#f87171"

def score_label(s):
    return "🚀 Excellent Candidate" if s > 75 else "👍 Good Candidate" if s > 50 else "⚠️ Needs Improvement"


st.markdown("<h1 style='text-align:center;margin-bottom:4px;'>🎯 AI Interview Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;margin-bottom:24px;'>Live Camera · Speech · Voice Confidence · All Simultaneous</p>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.3, 1], gap="large")
q_idx = st.session_state.q_index

with col_left:
    st.markdown("#### 📷 Live Camera Feed")

    ctx = webrtc_streamer(
        key="interview-cam",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    emotion_display = st.empty()

    if ctx.video_processor:
        emotion_display.markdown(
            f"<div class='emotion-live'>😊 Live Emotion: <b>{ctx.video_processor.get_emotion().upper()}</b></div>",
            unsafe_allow_html=True
        )

    if q_idx < len(QUESTIONS):
        st.markdown(
            f"<div class='question-box'>Q{q_idx+1} of {len(QUESTIONS)}: {QUESTIONS[q_idx]['question']}</div>",
            unsafe_allow_html=True
        )
        st.progress(q_idx / len(QUESTIONS))

with col_right:
    st.markdown("#### 🎤 Interview Controls")

    if st.session_state.stage == "idle" and q_idx < len(QUESTIONS):
        if ctx.state.playing:
            st.info("✅ Camera is live and tracking your emotion.\n\nClick **Start** — speak your answer, then speak 4 more seconds for voice analysis.")
            if st.button("▶ Start Answer Recording"):
                if ctx.video_processor:
                    ctx.video_processor.start_collecting()
                st.session_state.stage = "recording"
                st.rerun()
        else:
            st.warning("⚠️ Please click **START** on the camera feed first to enable your webcam.")

    elif st.session_state.stage == "recording":
        current_q = QUESTIONS[q_idx]

        st.markdown("### 🔴 Recording in Progress")
        st.info("🎤 Speak your answer clearly — up to 60 seconds.\n\n**Pause 2 seconds** when done. Then speak again 6 more seconds for voice confidence.")

        bucket = {"answer": None, "voice": None}
        t1 = threading.Thread(target=_speech_worker, args=(bucket,), daemon=True)
        t2 = threading.Thread(target=_voice_worker,  args=(bucket,), daemon=True)
        t1.start()
        t2.start()

        status_slot = st.empty()
        deadline = time.time() + 80   # 60s speech + 6s voice + buffer

        while (t1.is_alive() or t2.is_alive()) and time.time() < deadline:
            s1 = "⏳ listening..." if t1.is_alive() else "✅ done"
            s2 = "⏳ recording..." if t2.is_alive() else "✅ done"
            status_slot.markdown(f"**🎤 Speech:** {s1}   |   **🎙️ Voice:** {s2}")

            if ctx.video_processor:
                emotion_display.markdown(
                    f"<div class='emotion-live'>😊 Live Emotion: <b>{ctx.video_processor.get_emotion().upper()}</b></div>",
                    unsafe_allow_html=True
                )
            time.sleep(0.2)

        t1.join()
        t2.join()
        status_slot.empty()

        if ctx.video_processor:
            ctx.video_processor.stop_collecting()

        user_answer = bucket.get("answer") or ""
        voice_score = bucket.get("voice")  or "LOW (Nervous)"

        if not user_answer.strip():
            st.error("❌ No speech captured. Check your microphone and try again.")
            st.session_state.stage = "idle"
            st.rerun()

        with st.spinner("🧠 Evaluating answer..."):
            answer_score = evaluate_answer(user_answer, current_q["reference"])

        history      = ctx.video_processor.get_history() if ctx.video_processor else []
        final_emotion = dominant_emotion(history)
        score        = calc_score(answer_score, voice_score, final_emotion)

        st.session_state.pending_result = {
            "question":     current_q["question"],
            "answer":       user_answer,
            "answer_score": answer_score,
            "voice_score":  voice_score,
            "emotion":      final_emotion,
            "score":        score,
        }
        st.session_state.results.append(st.session_state.pending_result)
        st.session_state.q_index += 1
        st.session_state.stage   = "show_result"
        st.rerun()

    elif st.session_state.stage == "show_result" and st.session_state.pending_result:
        r = st.session_state.pending_result

        st.success(f"🗣 Your answer: *{r['answer']}*")
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Answer Quality</div><div class='metric-value'>{r['answer_score']}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Voice Confidence</div><div class='metric-value' style='font-size:0.9rem'>{r['voice_score']}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Dominant Emotion</div><div class='metric-value'>😊 {r['emotion']}</div></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card' style='margin-top:10px'>
            <div class='metric-label'>Question Score</div>
            <div class='score-ring' style='color:{score_color(r["score"])}'>{r['score']}<span style='font-size:1rem'>/100</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        if st.session_state.q_index < len(QUESTIONS):
            if st.button("▶ Next Question"):
                st.session_state.stage = "idle"
                st.session_state.pending_result = None
                st.rerun()
        else:
            if st.button("📊 See Final Results"):
                st.session_state.stage = "done"
                st.session_state.pending_result = None
                st.rerun()

    elif st.session_state.stage == "done" and st.session_state.results:
        avg = sum(r["score"] for r in st.session_state.results) // len(st.session_state.results)

        st.markdown("<h3 style='text-align:center'>🏆 Interview Complete</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-card' style='padding:30px'>
            <div class='metric-label'>Overall Score</div>
            <div class='score-ring' style='color:{score_color(avg)}'>{avg}<span style='font-size:1.2rem'>/100</span></div>
            <div style='color:{score_color(avg)};font-size:1.1rem;font-weight:600;margin-top:8px'>{score_label(avg)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Question Breakdown")
        for i, r in enumerate(st.session_state.results, 1):
            with st.expander(f"Q{i}: {r['question']}"):
                st.write(f"**Your Answer:** {r['answer']}")
                st.write(f"**Quality:** {r['answer_score']} | **Voice:** {r['voice_score']} | **Emotion:** {r['emotion']}")
                st.markdown(f"**Score: {r['score']}/100**")

        if st.button("🔄 Restart Interview"):
            for k, v in [("q_index",0), ("results",[]), ("stage","idle"), ("pending_result",None)]:
                st.session_state[k] = v
            st.rerun()