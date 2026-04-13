import os
import numpy as np
import torch
import joblib
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configurations
DATA_DIRS = {
    "positive": "../data/raw_audio/positive", # Label 1 (Wake Word)
    "near": "../data/raw_audio/near",         # Label 0 (Similar sounding words)
    "other": "../data/raw_audio/other"        # Label 0 (Background noise/random words)
}
MODEL_SAVE_PATH = "../data/models/wakeword_rf.joblib"

print("Loading Wav2Vec2 Foundation Model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval() # Set to inference mode

X = []
y = []

print("Extracting deep features from audio dataset (this may take a few minutes)...")
for category, path in DATA_DIRS.items():
    if not os.path.exists(path):
        print(f"Skipping {path}, directory not found.")
        continue
        
    label = 1 if category == "positive" else 0
    files = [f for f in os.listdir(path) if f.endswith(('.wav', '.m4a', '.mp3'))]
    
    print(f"Processing {len(files)} files in '{category}'...")
    for filename in files:
        file_path = os.path.join(path, filename)
        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            # Extract deep acoustic features
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Average the sequence to get a single, rich 768-Dimension vector
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

X = np.array(X)
y = np.array(y)

print("\nTraining Random Forest Classifier...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

print("\n--- Evaluation on 20% Holdout Set ---")
print(classification_report(y_test, clf.predict(X_test), target_names=["Sleep (0)", "Awake (1)"]))

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"\nSuccess! Wake word model saved to {MODEL_SAVE_PATH}")