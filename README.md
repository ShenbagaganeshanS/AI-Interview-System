AI-Based Interview Evaluation System
📌 Overview

The AI-Based Interview Evaluation System is a multi-modal machine learning application designed to automate candidate assessment during interviews. The system evaluates candidates based on three key aspects:

Answer Quality (Natural Language Processing)
Voice Confidence (Audio Analysis)
Facial Emotions (Computer Vision)

By combining these modalities, the system generates an objective interview performance score and provides real-time evaluation.

🚀 Features
🎤 Speech-to-Text Conversion
🧠 Semantic Answer Evaluation using Sentence-BERT
😊 Real-Time Facial Emotion Detection
🎙️ Voice Confidence Analysis using Pitch & Energy
🤖 Machine Learning-Based Score Prediction
📊 Candidate Performance Dashboard
🌐 Interactive Web Interface using Streamlit
📹 Real-Time Webcam Processing using WebRTC
🏗️ System Architecture
Candidate Answer
       │
       ▼
Speech-to-Text Conversion
       │
       ▼
Sentence-BERT Embedding
       │
       ▼
Semantic Similarity Score
       │
       ├──────────────┐
       │              │
       ▼              ▼
Voice Analysis    Emotion Detection
(Pitch & Energy)    (DeepFace)
       │              │
       └──────┬───────┘
              ▼
 Random Forest Regressor
              ▼
      Final Interview Score
🛠️ Technologies Used
Frontend
Streamlit
Streamlit-WebRTC
Natural Language Processing
Sentence-BERT
Cosine Similarity
Computer Vision
DeepFace
OpenCV
Speech Processing
SpeechRecognition
Google Speech API
Audio Processing
Librosa
NumPy
Machine Learning
Scikit-Learn
Random Forest Regressor
Backend
Python
📂 Project Structure
AI-Interview-System/
│
├── ui_app.py                 # Main Streamlit Application
├── speech_to_text.py         # Speech Recognition Module
├── voice.py                  # Voice Confidence Analysis
├── nlp.py                    # Semantic Similarity Evaluation
├── train_model.py            # Random Forest Training
├── score_model.pkl           # Trained ML Model
├── app.py                    # Emotion Detection Module
│
├── requirements.txt
└── README.md
⚙️ Working
Step 1: Speech Recognition

The candidate's spoken answer is converted into text using the SpeechRecognition library.

Step 2: Semantic Evaluation

Sentence-BERT converts both the candidate's answer and reference answer into embeddings. Cosine similarity is used to measure semantic similarity.

Step 3: Voice Analysis

Librosa extracts:

Energy
Pitch
Pitch Stability

These features are used to estimate confidence levels.

Step 4: Emotion Detection

DeepFace analyzes webcam frames and identifies dominant facial emotions such as:

Happy
Neutral
Sad
Angry
Surprise
Step 5: Final Score Prediction

Extracted features are passed to a Random Forest Regression model which predicts the final interview performance score.

📊 Evaluation Parameters
Parameter	Purpose
Semantic Similarity	Measures answer relevance
Voice Confidence	Measures speaking confidence
Facial Emotion	Measures behavioral response
Final Score	Overall interview performance
🎯 Advantages
Automated Interview Assessment
Objective Candidate Evaluation
Real-Time Analysis
Multi-Modal Learning Approach
Reduced Human Bias
Scalable for Recruitment Platforms
🔮 Future Enhancements
Integration with Large Language Models (LLMs)
AI-Based Personalized Feedback
Support for Multiple Languages
Advanced Emotion Recognition Models
Interview Performance Analytics Dashboard
Cloud Deployment for Enterprise Use
📈 Results

The system successfully combines NLP, Computer Vision, Audio Signal Processing, and Machine Learning to evaluate candidates in real time and generate an overall interview performance score.

👨‍💻 Author

Shenbaga Ganeshan

AI-Based Interview Evaluation System
Machine Learning | NLP | Computer Vision | Audio Analytics | Streamlit Development