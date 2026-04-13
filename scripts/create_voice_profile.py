import os
import torch
import numpy as np
import functools
from pydub import AudioSegment
from speechbrain.inference.speaker import EncoderClassifier

# Monkey-patch for PyTorch 2.x compatibility
# torch.amp.custom_fwd was removed in PyTorch 2.0, replace with identity wrapper
if not hasattr(torch.amp, 'custom_fwd'):
    def custom_fwd_noop(func, device_type=None, cast_inputs=None):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    torch.amp.custom_fwd = custom_fwd_noop

if not hasattr(torch.amp, 'custom_bwd'):
    def custom_bwd_noop(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    torch.amp.custom_bwd = custom_bwd_noop

# Configurations
DATA_DIR = "data/raw_audio/positive"
PROFILE_SAVE_PATH = "data/models/team_voice_profiles.npz" # Note the .npz extension!
TEAM_NAMES = ["nandal", "kubde", "khodabakhsh"]

print("Downloading/Loading Pre-Trained SpeechBrain Model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="data/models/ecapa"
)

# Create a dictionary to hold each person's list of fingerprints
user_embeddings = {name: [] for name in TEAM_NAMES}

print("Extracting individual vocal fingerprints...")
for filename in os.listdir(DATA_DIR):
    lower_filename = filename.lower()
    
    # Figure out whose voice this is based on the filename
    current_speaker = None
    for name in TEAM_NAMES:
        if name in lower_filename:
            current_speaker = name
            break
            
    if current_speaker and filename.endswith(('.wav', '.m4a', '.mp3')):
        file_path = os.path.join(DATA_DIR, filename)
        
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        signal = torch.tensor(samples).unsqueeze(0)
        
        with torch.no_grad():
            emb = classifier.encode_batch(signal)
            user_embeddings[current_speaker].append(emb.squeeze().cpu().numpy())

# Average each person's voice individually
master_keyring = {}
for name, embs in user_embeddings.items():
    if not embs:
        print(f"Warning: No audio files found for {name}!")
        continue
    
    master_keyring[name] = np.mean(embs, axis=0)
    print(f"Processed {len(embs)} samples for {name}.")

if not master_keyring:
    raise ValueError("No valid audio files processed!")

os.makedirs(os.path.dirname(PROFILE_SAVE_PATH), exist_ok=True)
# Save as a dictionary of arrays
np.savez(PROFILE_SAVE_PATH, **master_keyring)

print(f"Success! Team Voice dict saved to {PROFILE_SAVE_PATH}")