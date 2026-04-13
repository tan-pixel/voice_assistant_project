import asyncio
import edge_tts
import os
import pygame

class TTSModule:
    def __init__(self):
        # We use JennyNeural as a clean, default voice
        # can change this to 'en-CA-ClaraNeural' or others
        self.voice = "en-US-JennyNeural"
        self.output_file = "data/tts_output.mp3"
        self.audio_enabled = True
        
        # Initialize the pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except Exception as e:
            self.audio_enabled = False
            print(f"TTS audio playback disabled: {e}")
        
        os.makedirs("data", exist_ok=True)

    def process(self, text):
        """
        Input: Natural Language string from M6.
        Output: Plays the audio and returns a success state.
        """
        if not text or text == "Audio skipped":
            return "No text to speak"

        print(f"Generating audio for: '{text}'")

        if not hasattr(edge_tts, "Communicate"):
            print("TTS skipped: edge_tts.Communicate is unavailable in this environment.")
            return "Audio skipped"

        try:
            # Run the async edge-tts function synchronously
            asyncio.run(self._generate_audio(text))

            # Play the generated audio file
            self._play_audio()
            return "Audio playback complete"
        except Exception as e:
            print(f"TTS generation failed: {e}")
            return "Audio skipped"
        
    async def _generate_audio(self, text):
        """Internal async method to call the Microsoft Edge TTS API."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(self.output_file)
        
    def _play_audio(self):
        """Plays the mp3 file and waits for it to finish."""
        if not self.audio_enabled:
            print("Audio playback skipped: mixer is unavailable.")
            return

        try:
            pygame.mixer.music.load(self.output_file)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing before returning control to the pipeline
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            # Unload the file so it can be overwritten on the next turn
            pygame.mixer.music.unload()
            
        except Exception as e:
            print(f"Error playing audio: {e}")
