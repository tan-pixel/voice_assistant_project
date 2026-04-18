import os
import torch
import sounddevice as sd
from kokoro import KPipeline
import asyncio

# Dorsa's Fallback Imports
import pygame
try:
    import edge_tts
except ImportError:
    edge_tts = None

class TTSModule:
    def __init__(self):
        # Kokoro Initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading M7: Kokoro-TTS on {self.device} (Primary)...")
        self.use_local = True
        
        try:
            self.pipeline = KPipeline(lang_code='a', device=self.device)
            self.kokoro_voice = 'am_santa' 
        except Exception as e:
            print(f"Warning: Kokoro failed to initialize ({e}). Defaulting to Cloud TTS.")
            self.use_local = False

        # Edge-TTS & Pygame (Fallback)
        print("Loading M7: Edge-TTS (Fallback)...")
        self.edge_voice = "en-US-JennyNeural"
        self.output_file = "data/tts_output.mp3"
        self.audio_enabled = True
        
        try:
            pygame.mixer.init()
        except Exception as e:
            self.audio_enabled = False
            print(f"Pygame audio playback disabled: {e}")
        
        os.makedirs("data", exist_ok=True)

    def process(self, text):
        """
        Input: Natural language string from M6
        Output: Plays high-fidelity audio directly to the speakers
        """
        if not text:
            return

        print(f"[M7 TTS] Generating: '{text}'")

        # Kokoro Engine
        if self.use_local:
            try:
                generator = self.pipeline(
                    text, 
                    voice=self.kokoro_voice, 
                    speed=1.0, 
                    split_pattern=r'\n+'
                )
                for i, (graphemes, phonemes, audio) in enumerate(generator):
                    sd.play(audio, samplerate=24000)
                    sd.wait() 
                return "Audio playback complete"
                
            except Exception as e:
                print(f"[M7 Error] Kokoro TTS failed: {e}. Switching to Fallback Engine...")

        # If Kokoro fails (or is disabled), use Edge-TTS Fallback
        if edge_tts is None or not hasattr(edge_tts, "Communicate"):
            print("TTS skipped: edge_tts is unavailable in this environment.")
            return "Audio skipped"

        try:
            asyncio.run(self._generate_audio(text))
            self._play_audio()
            return "Audio playback complete"
        except Exception as e:
            print(f"[M7 Error] Edge-TTS generation failed: {e}")
            return "Audio skipped"
        
    async def _generate_audio(self, text):
        """Internal async method to call the Microsoft Edge TTS API."""
        communicate = edge_tts.Communicate(text, self.edge_voice)
        await communicate.save(self.output_file)
        
    def _play_audio(self):
        """Plays the mp3 file and waits for it to finish."""
        if not self.audio_enabled:
            print("Audio playback skipped: mixer is unavailable.")
            return
            
        try:
            pygame.mixer.music.load(self.output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Error playing Edge-TTS audio: {e}")