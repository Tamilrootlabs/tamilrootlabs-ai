"""
Pro Voice Assistant v2
Pipeline: Microphone -> Groq Whisper (STT) -> Groq LLM -> edge-tts (TTS) -> Speakers

Setup:
    pip install SpeechRecognition pyaudio pygame groq edge-tts python-dotenv

    .env file (add it to .gitignore):
        GROQ_API_KEY=your_key_here

    Optional .env overrides: LLM_MODEL, STT_MODEL, TTS_VOICE

Privacy note: microphone audio is sent to Groq for transcription.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import edge_tts
import pygame
import speech_recognition as sr
from groq import Groq

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  
    pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("voice_assistant")


# Configuration
@dataclass(frozen=True)
class Config:
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    stt_model: str = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
    tts_voice: str = os.getenv("TTS_VOICE", "en-US-AriaNeural")
    language: str = "en"

    # Audio capture
    calibration_seconds: float = 1.5
    listen_timeout: float = 6.0
    phrase_time_limit: float = 15.0
    pause_threshold: float = 0.8
    min_audio_seconds: float = 0.4  
    sample_rate: int = 16000

    # Speech filtering
    no_speech_threshold: float = 0.6
    min_text_chars: int = 2

    # LLM / network
    max_history_messages: int = 12
    max_output_tokens: int = 600 
    request_timeout: float = 25.0
    max_retries: int = 2
    max_consecutive_errors: int = 5
    system_prompt: str = (
        "You are a friendly voice assistant. Always reply in English only. "
        "Keep answers to one or two short sentences. Your reply will be spoken "
        "aloud, so do not use emojis, markdown, lists, or special formatting."
    )

    exit_phrases: frozenset = field(
        default_factory=lambda: frozenset(
            {"exit", "quit", "goodbye", "bye", "stop", "shut down"}
        )
    )
    reset_phrases: frozenset = field(
        default_factory=lambda: frozenset(
            {"reset", "forget everything", "new conversation", "clear memory"}
        )
    )


def load_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        sys.exit(
            "GROQ_API_KEY is not set.\n"
            "Add it to a .env file or your environment. Never hardcode it."
        )
    return key



# Text helpers
_MARKDOWN_RE = re.compile(r"[*_`#>~]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F2FF"
    "\uFE0F\u200D"
    "]+",
    flags=re.UNICODE,
)
_SPACES_RE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return _SPACES_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# Modules
class Ear:
    def __init__(self, cfg: Config, client: Groq):
        self.cfg = cfg
        self.client = client

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = cfg.pause_threshold
        self.recognizer.dynamic_energy_threshold = True

        self.mic = sr.Microphone(sample_rate=cfg.sample_rate)
        self._calibrate()

    def _calibrate(self) -> None:
        print("Calibrating microphone, stay quiet for a moment...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(
                source, duration=self.cfg.calibration_seconds
            )

    def listen(self) -> str:
        with self.mic as source:
            print("\nListening... speak now.")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.cfg.listen_timeout,
                    phrase_time_limit=self.cfg.phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                print("No speech detected.")
                return ""

        wav_bytes = audio.get_wav_data()
        duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
        if duration < self.cfg.min_audio_seconds:
            return ""

        print("Processing speech...")
        try:
            result = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=self.cfg.stt_model,
                language=self.cfg.language,
                response_format="verbose_json",
                temperature=0.0,
            )
        except Exception:
            log.exception("Transcription failed")
            return ""

        return self._filter(result)

    def _filter(self, result: Any) -> str:
        segments = _get(result, "segments") or []
        if segments:
            probs = [_get(s, "no_speech_prob", 0.0) or 0.0 for s in segments]
            if sum(probs) / len(probs) > self.cfg.no_speech_threshold:
                return ""

        text = (_get(result, "text", "") or "").strip()
        return text if len(text) >= self.cfg.min_text_chars else ""


class Brain:
    FALLBACK = "Sorry, I am having trouble connecting right now."

    def __init__(self, cfg: Config, client: Groq):
        self.cfg = cfg
        self.client = client
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def _trim(self) -> None:
        limit = self.cfg.max_history_messages
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def think(self, user_text: str) -> str:
        if not user_text:
            return ""

        print("Thinking...")
        self.history.append({"role": "user", "content": user_text})
        self._trim()
        messages = [{"role": "system", "content": self.cfg.system_prompt}, *self.history]

        try:
            response = self.client.chat.completions.create(
                model=self.cfg.llm_model,
                messages=messages,
                max_completion_tokens=self.cfg.max_output_tokens,
                temperature=0.6,
                reasoning_effort="low",
            )
            reply = (response.choices[0].message.content or "").strip()
        except Exception:
            log.exception("LLM request failed")
            self.history.pop()  
            return self.FALLBACK

        if not reply:
            self.history.pop()
            return "Sorry, I did not get a response. Could you say that again?"

        self.history.append({"role": "assistant", "content": reply})
        self._trim()
        return reply


class Mouth:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        pygame.mixer.init()

    async def _synthesize(self, text: str, path: Path) -> None:
        await edge_tts.Communicate(text, self.cfg.tts_voice).save(str(path))

    def speak(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return

        print("Speaking...")
        with tempfile.TemporaryDirectory(prefix="va_tts_") as tmp:
            audio_path = Path(tmp) / "reply.mp3"
            try:
                asyncio.run(self._synthesize(text, audio_path))
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.play()
                clock = pygame.time.Clock()
                while pygame.mixer.music.get_busy():
                    clock.tick(10)
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Speech playback failed")
            finally:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()  
                except Exception:
                    pass

    def close(self) -> None:
        try:
            pygame.mixer.quit()
        except Exception:
            pass


# Orchestrator
class VoiceAssistant:
    def __init__(self, cfg: Config, api_key: str):
        print("Initializing AI systems, please wait...")
        self.cfg = cfg

        client = Groq(
            api_key=api_key,
            timeout=cfg.request_timeout,
            max_retries=cfg.max_retries,
        )
        self.ear = Ear(cfg, client)
        self.brain = Brain(cfg, client)
        self.mouth = Mouth(cfg)
        print("System ready.\n" + "-" * 50)

    def _handle_turn(self) -> bool:
        user_text = self.ear.listen()
        if not user_text:
            return True
        print(f"You: {user_text}")

        cmd = normalize(user_text)
        if cmd in self.cfg.exit_phrases:
            self.mouth.speak("Goodbye! Have a great day.")
            return False
        if cmd in self.cfg.reset_phrases:
            self.brain.reset()
            self.mouth.speak("Okay, I have cleared our conversation.")
            return True

        reply = self.brain.think(user_text)
        if reply:
            print(f"AI: {reply}")
            self.mouth.speak(reply)
        return True

    def run(self) -> None:
        print("Voice assistant started.")
        print("Press Enter to talk. Say 'goodbye' or press Ctrl+C to quit.")

        errors = 0
        try:
            while True:
                input("\nPress Enter to start...")
                try:
                    if not self._handle_turn():
                        break
                    errors = 0
                except KeyboardInterrupt:
                    raise
                except Exception:
                    errors += 1
                    log.exception("Unexpected error during turn")
                    if errors >= self.cfg.max_consecutive_errors:
                        print("Too many consecutive errors, shutting down.")
                        break
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down safely.")
        finally:
            self.mouth.close()


def main() -> None:
    VoiceAssistant(Config(), load_api_key()).run()


if __name__ == "__main__":
    main()