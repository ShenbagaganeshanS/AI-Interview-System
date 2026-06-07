import sounddevice as sd
import numpy as np
import librosa


def analyze_voice(duration=5, fs=22050):
    """
    Records audio for `duration` seconds and analyzes voice confidence
    based on energy and pitch stability.
    Returns: "HIGH (Confident)", "MEDIUM", or "LOW (Nervous)"
    """
    print(f"🎙️ Recording voice for {duration} seconds... Speak now!")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
    except Exception as e:
        print("Recording error:", e)
        return "LOW (Nervous)"

    energy = np.mean(np.square(audio))

    try:
        pitch_values = librosa.yin(audio, fmin=50, fmax=300)
        # Filter out unvoiced frames (pitch=0 or very low values)
        voiced = pitch_values[pitch_values > 60]
        if len(voiced) > 0:
            pitch_mean = np.mean(voiced)
            pitch_std = np.std(voiced)
        else:
            pitch_mean = 0
            pitch_std = 100
    except Exception as e:
        print("Pitch analysis error:", e)
        pitch_mean = 0
        pitch_std = 100

    print(f"  Energy: {energy:.5f}")
    print(f"  Pitch Mean: {pitch_mean:.2f} Hz")
    print(f"  Pitch Variation (std): {pitch_std:.2f}")

    
    if energy > 0.004 and pitch_std < 70:
        return "HIGH (Confident)"
    elif energy > 0.002 and pitch_std < 90:
        return "MEDIUM"
    else:
        return "LOW (Nervous)"


if __name__ == "__main__":
    result = analyze_voice()
    print("Confidence Level:", result)