import os
import torch
import soundfile as sf
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cosine

class UserVerificationModule:
    def __init__(self):
        self.profile_path = "data/models/team_voice_profiles.npz"
        
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Keyring not found at '{self.profile_path}'. Run the setup script.")
            
        print("Loading Pre-Trained Verification Model...")
        # Load the dictionary of profiles
        self.master_keyring = np.load(self.profile_path)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="data/models/ecapa"
        )
        
        # We can tweak the threshold to make it more/less strict
        self.similarity_threshold = 0.35

    def process(self, audio_path):
        """
        Input: Path to the user's recorded voice
        Output: 'Unlocked' if verified, 'Locked' otherwise.
        """
        try:
            data, sr = sf.read(audio_path)
            
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                
            signal = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                live_emb = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
            
            # Compare the live voice against EVERY person in the keyring
            best_score = 0.0
            best_match = "Unknown"
            
            for name in self.master_keyring.files:
                profile_emb = self.master_keyring[name]
                similarity = 1 - cosine(profile_emb, live_emb)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = name
                    
            print(f"[M1 Verification] Best Match: {best_match} (Score: {best_score:.2f} / Threshold: {self.similarity_threshold})")
            
            if best_score >= self.similarity_threshold:
                # We can even return the name if we want Atlas to greet us personally
                return "Unlocked" 
            else:
                return "Locked"
                
        except Exception as e:
            print(f"[M1 Error] Verification failed: {e}")
            return "Locked"