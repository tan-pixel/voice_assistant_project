import torch
import sounddevice as sd
from kokoro import KPipeline

class TTSModule:
    def __init__(self):
        # Determine Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading M7: Kokoro-TTS on {self.device}...")

        # Initialize the Pipeline ('a' = American English, 'b' = British English)
        self.pipeline = KPipeline(lang_code='a', device=self.device)
        
        # Lock the Voice Profile
        self.voice = 'af_heart' 

    def process(self, text):
        """
        Input: Natural language string from M6
        Output: Plays high-fidelity audio directly to the speakers
        """
        if not text:
            return

        print(f"[M7 TTS] Generating: '{text}'")
        try:
            # Kokoro generates audio as a generator, reading chunk by chunk
            generator = self.pipeline(
                text, 
                voice=self.voice, 
                speed=1.0, 
                split_pattern=r'\n+' # Splits text by line breaks to manage memory
            )

            # Iterate through the generated audio chunks and play them immediately
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                sd.play(audio, samplerate=24000)
                sd.wait() 
                
        except Exception as e:
            print(f"[M7 Error] Kokoro TTS failed: {e}")