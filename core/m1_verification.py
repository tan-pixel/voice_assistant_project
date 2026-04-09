import os
from tensorflow.keras.models import load_model
from utils.audio_utils import extract_mfcc_for_inference

class UserVerificationModule:
    def __init__(self):
        self.model_path = "data/models/verification_model.keras"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Verification model not found at '{self.model_path}'. Please run the training script.")
            
        self.model = load_model(self.model_path)

    def process(self, audio_path):
        """
        Input: Path to the user's recorded voice.
        Output: 'Unlocked' if verified, 'Locked' otherwise.
        """
        # Extract features
        mfcc_features = extract_mfcc_for_inference(audio_path)
        
        # Predict (Binary classification: 1 = Team Member, 0 = Other)
        prediction = self.model.predict(mfcc_features, verbose=0)[0][0]
        
        if prediction > 0.5:
            return "Unlocked"
        else:
            return "Locked"