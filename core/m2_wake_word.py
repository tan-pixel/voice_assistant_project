import os
from tensorflow.keras.models import load_model
from utils.audio_utils import extract_mfcc_for_inference

class WakeWordDetectionModule:
    def __init__(self):
        self.model_path = "data/models/wakeword_model.h5"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Wake word model not found at '{self.model_path}'. Please run the training script.")
            
        self.model = load_model(self.model_path)

    def process(self, audio_path):
        """
        Input: Path to the user's recorded voice.
        Output: 'Awake' if wake word detected, 'Sleep' otherwise.
        """
        # Extract features
        mfcc_features = extract_mfcc_for_inference(audio_path)
        
        # Predict (Binary classification: 1 = Wake Word, 0 = Near/Other)
        prediction = self.model.predict(mfcc_features, verbose=0)[0][0]
        
        if prediction > 0.5:
            return "Awake"
        else:
            return "Sleep"