import speech_recognition as sr
import time


def get_speech_input(timeout=30, phrase_limit=60) -> str:
    """
    Listens to microphone input and converts speech to text.

    Improvements over basic version:
    - Longer timeout (30s) and phrase limit (60s) for full interview answers
    - Dynamic energy threshold (auto-adjusts to your mic and room noise)
    - Longer pause_threshold so natural speaking pauses don't cut you off
    - Non_speaking_duration tuned so silence detection is less aggressive
    - Retries up to 2 times on UnknownValueError before giving up
    - Falls back to offline Sphinx recognition if Google API fails
    - Audio is re-recorded fresh each retry (not re-submitted stale audio)

    Returns: recognized text string, or empty string on total failure.
    """
    recognizer = sr.Recognizer()

    # ── Tuned recognizer settings ───────────────────────────────
    recognizer.dynamic_energy_threshold      = True   # auto-adjust to room noise
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_ratio          = 1.5
    recognizer.energy_threshold              = 200    # low starting point; dynamic will raise it
    recognizer.pause_threshold               = 2.0    # 2 sec of silence = end of phrase (was 1.0)
    recognizer.non_speaking_duration         = 1.5    # silence window for phrase boundary
    recognizer.phrase_threshold              = 0.3    # minimum seconds of speaking to count

    max_retries = 2

    for attempt in range(1, max_retries + 1):
        print(f"\n🎤 Attempt {attempt}/{max_retries}")
        try:
            with sr.Microphone() as source:
                # Calibrate for ambient noise on every attempt
                print("🔇 Calibrating for background noise (1.5 sec)...")
                recognizer.adjust_for_ambient_noise(source, duration=1.5)
                print(f"🎤 Speak your answer now... (up to {phrase_limit} seconds)")
                print("   (Pause for 2 seconds when done speaking)\n")

                audio = recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )

            # ── Primary: Google Web Speech API ──────────────────
            # language="en-IN" improves accuracy for Indian-accented English.
            # Change to "en-US" or "en-GB" if needed.
            try:
                text = recognizer.recognize_google(
                    audio,
                    language="en-IN",
                    show_all=False
                )
                if text.strip():
                    print(f"✅ Recognized: {text}")
                    return text.strip()
                else:
                    print("⚠️ Empty result from Google, retrying...")
                    continue

            except sr.UnknownValueError:
                print(f"⚠️ Google could not understand audio (attempt {attempt})")
                if attempt < max_retries:
                    print("   Retrying — please speak clearly and a bit louder...")
                    time.sleep(0.5)
                    continue

            except sr.RequestError as e:
                print(f"⚠️ Google API unavailable: {e}")
                # ── Fallback: try offline Sphinx ─────────────────
                try:
                    print("🔄 Trying offline Sphinx fallback...")
                    text = recognizer.recognize_sphinx(audio)
                    if text.strip():
                        print(f"✅ Sphinx recognized: {text}")
                        return text.strip()
                except Exception as sphinx_err:
                    print(f"⚠️ Sphinx also failed: {sphinx_err}")
                break  # No point retrying if network is down

        except sr.WaitTimeoutError:
            print(f"⚠️ No speech detected within {timeout} seconds.")
            if attempt < max_retries:
                print("   Please make sure your microphone is working and try again.")
                time.sleep(0.5)
            continue

        except OSError as e:
            print(f"⚠️ Microphone device error: {e}")
            print("   Check that your microphone is connected and not in use by another app.")
            break

        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            break

    print("❌ Speech recognition failed after all attempts.")
    return ""


if __name__ == "__main__":
    print("=== Speech-to-Text Test ===")
    result = get_speech_input()
    if result:
        print(f"\n📝 Final captured text:\n{result}")
    else:
        print("\n❌ Nothing captured.")