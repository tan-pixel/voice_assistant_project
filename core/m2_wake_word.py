import os
import torch
import soundfile as sf
import numpy as np
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class WakeWordDetectionModule:
    def __init__(self):
        self.model_path = "data/models/wakeword_rf.joblib"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Wake word model not found at '{self.model_path}'. Run the RF training script.")
            
        print("Loading M2: Wav2Vec2 + Random Forest...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.w2v_model.eval()
        
        # Load our trained Scikit-Learn classifier
        self.clf = joblib.load(self.model_path)

    def process(self, audio_path):
        """
        Input: Path to the user's recorded voice (.wav)
        Output: 'Awake' if wake word detected, 'Sleep' otherwise.
        """
        try:
            data, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                
            # Extract Deep Features
            inputs = self.processor(data, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = self.w2v_model(**inputs)
                
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Predict 
            features_2d = features.reshape(1, -1)
            prediction = self.clf.predict(features_2d)[0]
            
            # Get the confidence percentage just for cool debugging logs
            probabilities = self.clf.predict_proba(features_2d)[0]
            confidence = max(probabilities) * 100
            
            print(f"[M2 Wake Word] Prediction: {'Awake' if prediction == 1 else 'Sleep'} | Confidence: {confidence:.1f}%")
            
            if prediction == 1:
                return "Awake"
            else:
                return "Sleep"
                
        except Exception as e:
            print(f"[M2 Error] Wake word detection failed: {e}")
            return "Sleep"