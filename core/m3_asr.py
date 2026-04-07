import torch 
import whisper
import warnings

class ASRModule:
    def __init__(self):
        warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper ASR model on {self.device}...")
        
        self.model = whisper.load_model("base", device=self.device)

    def process(self, audio_path):
        """
        Input: Path to the user's recorded voice.
        Output: Transcribed English text string.
        """
        result = self.model.transcribe(
            audio_path,
            language="en", 
            fp16=(self.device == "cuda"), # float16 for a massive speed boost on GPU
            verbose=False
        )
        
        # Clean and return the transcribed text
        transcription = result["text"].strip()
        
        # Remove trailing punctuation that might confuse the Intent BERT model
        transcription = transcription.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        
        return transcription