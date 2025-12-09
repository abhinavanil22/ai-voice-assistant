import os
import time
from pathlib import Path
import wave

import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound

from google import genai
from google.genai import types

# ---------- CONFIG ----------

# Uses GEMINI_API_KEY from environment
client = genai.Client()

# Models
TEXT_MODEL = "gemini-2.5-flash"              # main chat + transcription model
TTS_MODEL = "gemini-2.5-flash-preview-tts"   # text-to-speech model

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)


# ---------- AUDIO RECORDING ----------

def record_audio(filename="input.wav", duration=5, fs=44100):
    """
    Record audio from microphone for `duration` seconds and save as WAV.
    """
    filepath = AUDIO_DIR / filename
    print(f"\n[INFO] Recording for {duration} seconds... Speak now!")
    time.sleep(0.5)

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished

    write(filepath, fs, recording)
    print(f"[INFO] Recording saved to {filepath}")
    return filepath


# ---------- SPEECH TO TEXT (Gemini) ----------

def speech_to_text(wav_path: Path) -> str:
    """
    Send recorded audio to Gemini and return transcribed text.
    Uses audio understanding with inline audio bytes.
    """
    print("[INFO] Transcribing audio with Gemini...")

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    # Prompt Gemini to do pure transcription
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[
            # Instruction
            "Transcribe this speech. Return ONLY what the user said as plain text. "
            "Do not add extra words or explanations.",
            # Audio as inline bytes
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav",   # WAV from sounddevice/scipy
            ),
        ],
    )

    text = response.text.strip()
    print(f"[USER SAID] {text}")
    return text


# ---------- LLM BRAIN (Gemini) ----------

def build_conversation_prompt(user_text: str, history: list) -> str:
    """
    Build a simple chat-style prompt from history + latest user text.
    History is a list of (user, assistant) pairs.
    """
    intro = (
        "You are Abhinav's personal voice assistant.\n"
        "- You are friendly and concise.\n"
        "- Abhinav is a CS student and beginner programmer.\n"
        "- Explain things clearly and practically.\n"
        "- Keep answers short unless the user asks for details.\n\n"
    )

    convo = intro
    for u, a in history:
        convo += f"User: {u}\nAssistant: {a}\n\n"

    convo += f"User: {user_text}\nAssistant:"
    return convo


def ask_llm(user_text: str, history: list) -> str:
    """
    Send user text + history to Gemini and get reply.
    """
    prompt = build_conversation_prompt(user_text, history)

    print("[INFO] Asking Gemini...")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
    )

    assistant_text = response.text.strip()
    print(f"[ASSISTANT] {assistant_text}")
    return assistant_text


# ---------- TEXT TO SPEECH (Gemini TTS) ----------

def write_wav(filename: Path, pcm: bytes, channels=1, rate=24000, sample_width=2):
    """
    Save raw PCM bytes to a .wav file.
    Gemini TTS returns 24kHz mono PCM by default. :contentReference[oaicite:2]{index=2}
    """
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def text_to_speech(text: str, filename="reply.wav") -> Path:
    """
    Convert text to speech using Gemini TTS and save as WAV.
    """
    speech_path = AUDIO_DIR / filename
    print("[INFO] Generating speech audio with Gemini TTS...")

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # choose any voice you like
                    )
                )
            ),
        ),
    )

    # PCM bytes from inline_data
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    write_wav(speech_path, pcm_data)

    print(f"[INFO] Speech saved to {speech_path}")
    return speech_path


def play_audio(path: Path):
    """
    Play an audio file (blocking).
    """
    print("[INFO] Playing response...")
    playsound(str(path))


# ---------- MAIN LOOP ----------

def main():
    print("=======================================")
    print("   Abhinav's Gemini Voice Assistant")
    print("=======================================")
    print("Controls:")
    print("  - Press ENTER to start recording (5 seconds).")
    print("  - Type 'q' and press ENTER to quit.")
    print("---------------------------------------")

    history = []  # list of (user_text, assistant_text)

    while True:
        cmd = input("\nPress ENTER to talk, or 'q' + ENTER to quit: ").strip().lower()
        if cmd == "q":
            print("Goodbye! 👋")
            break

        # 1) Record
        wav_path = record_audio()

        # 2) STT
        try:
            user_text = speech_to_text(wav_path)
        except Exception as e:
            print(f"[ERROR] Speech-to-text (Gemini) failed: {e}")
            continue

        if user_text.strip() == "":
            print("[WARN] Got empty text, try speaking more clearly.")
            continue

        # 3) LLM
        try:
            assistant_text = ask_llm(user_text, history)
        except Exception as e:
            print(f"[ERROR] Gemini text generation failed: {e}")
            continue

        # Add to history for context
        history.append((user_text, assistant_text))

        # 4) TTS + play
        try:
            wav_reply = text_to_speech(assistant_text)
            play_audio(wav_reply)
        except Exception as e:
            print(f"[ERROR] Text-to-speech / playback failed: {e}")
            continue


if __name__ == "__main__":
    main()
