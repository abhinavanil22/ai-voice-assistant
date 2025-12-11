import os
import time
from pathlib import Path
import wave
import json
import webbrowser
import subprocess
from datetime import datetime
from dotenv import load_dotenv



import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound

from google import genai
from google.genai import types

# ---------- CONFIG ----------

load_dotenv()  # load .env

GEMINI_KEY = os.getenv("GEMINI_API_KEY") # Read Gemini API key from .env

# Create Gemini client using the key
client = genai.Client(api_key=GEMINI_KEY)

# Models
TEXT_MODEL = "gemini-2.5-flash"              # main chat + transcription model
TTS_MODEL = "gemini-2.5-flash-preview-tts"   # text-to-speech model

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

NOTES_FILE = Path("notes.json")


# ---------- NOTES (MEMORY) ----------


def load_notes():
    if not NOTES_FILE.exists():
        return []
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)




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


# ---------- COMMANDS & MEMORY LOGIC ----------

def handle_commands_and_memory(user_text: str, notes: list):
    """

    Delete and handle:
        open youtube
        google/search google 
        open vs code
        remember
        show my notes / what did you remenmber
    Returns (handled: bool, assistant_reply: st)
    """
    text = user_text.strip()
    lower = text.lower()

    # --- Memory: remember ---
    if lower.startswith("remenmber that "):
        note_body + text[len("remember that "):].strip()
    elif lower.startswith("remember "):
        note_body = text[len("remember "):].strip()
    else:
        note_body = None
    
    if note_body:
        note = {
            "text": note_body,
            "text": daterime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        notes.append(note)
        save_notes(notes)
        reply = "Okay, I will remember that."
        return True, reply

    # --- Memory: show notes / what did you remember ---
    if(
        "show my notes" in lower
        or "what did you remember" in lower
        or "what notes do you have" in lower
        or "show notes" in lower
    ):
        if not notes:
            reply = "Idon't have notes saved yet."
        else:
            reply_lines = ["Here are the notes I have:"]
            for i, n in enumerate(notes, start=1):
                reply_lines.append(f"{i}.{n['text']}")
            reply = " ".join(reply_lines)
        return True, reply


    # --- command: open youtube ---
    if "open youtube" in lower:
        webbrowser.open("https://www.youtube.com")
        reply = "Opening Youtube in your browser."
        return True, reply

    # --- command: google search ---
    if "search google for" in lower:
        idx = lower.find("search google ofr")
        query = text[idx + len("search google for"):].strip()
    elif "google for" in lower:
        idx = lower.find("google for")
        query = text[idx + len("google for"):].strip()
    else:
        query = None

    if query:
        url = "htts://www.google.com/search?q="+ query.replace(" ", "+")
        webbrowser.open(url)
        reply = f"Searching google for {query}."
        return True, reply

    # --- command: open vs code ---
    if "open vs code" in lower or "open vscode" in lower or "open visual studio code" in lower:
        try:
            #assumes 'code' is in PATH
            sunbprocess.Popen(["code"])
            reply = "opening Visual Studio Code..."
        except Exception:
            reply = "Cannot open VS Code. It seems the 'code' command is not available."
        return True, reply

    #nothing special handled
    return False, ""




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
    notes = load_notes()

    while True:
        cmd = input("\nPress ENTER to talk, or 'q' + ENTER to quit: ").strip().lower()
        if cmd == "q":
            print("Goodbye!!!")
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

        # 3) check command or memory
        handled,assistant_text = handle_commands_and_memory(user_text, notes)

        if handled:
            #special command or memory action was executed
            print(f"[ASSISTANT - COMMAND/MEMORY] {assistant_text}")
        else:
             # 4) LLM
            try:
                assistant_text = ask_llm(user_text, history)
            except Exception as e:
                print(f"[ERROR] Gemini text generation failed: {e}")
                continue        
         


        # Add to history for context
        history.append((user_text, assistant_text))

        # 5) TTS + play
        try:
            wav_reply = text_to_speech(assistant_text)
            play_audio(wav_reply)
        except Exception as e:
            print(f"[ERROR] Text-to-speech / playback failed: {e}")
            continue


if __name__ == "__main__":
    main()
