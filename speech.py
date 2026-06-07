from speech_to_text import get_speech_input
from nlp import evaluate_answer

reference = "Machine learning is a method where models learn from data."

print("🎤 Speak your answer...")
user_answer = get_speech_input()

if not user_answer.strip():
    print("❌ No speech captured.")
else:
    result = evaluate_answer(user_answer, reference)
    print("Answer Quality:", result)