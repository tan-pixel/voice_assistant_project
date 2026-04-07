import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

def extract_mfcc_for_inference(file_path, sr=16000, duration=2.0, n_mfcc=13, window_sec=0.025, hop_sec=0.025):
    """
    Loads an audio file, standardizes its length, and extracts MFCC features
    formatted for a Keras CNN input (batch_size, time_steps, mfcc_coeffs, channels).
    """
    # Load audio
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    # Calculate target samples
    num_samples = int(sr * duration)

    # Pad or truncate
    if len(y) < num_samples:
        y = np.pad(y, (0, num_samples - len(y)))
    else:
        y = y[:num_samples]

    # Calculate FFT and Hop lengths
    n_fft = int(window_sec * sr)
    hop_length = int(hop_sec * sr)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Add batch and channel dimensions for Keras: shape becomes (1, mfcc_coeffs, time_steps, 1)
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc_input

def record_audio(file_path="data/live_input.wav", duration=3.0, sr=16000):
    """Records live audio from the microphone and saves it to a file."""
    print(f"Listening for {duration} seconds...")

    # Set the default input device
    # Change this index if you have multiple audio input devices and want to specify which one to use (check using print(sd.query_devices()))
    # 2 works for my system
    sd.default.device[0] = 2

    # Record audio
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    
    # Save as WAV file
    sf.write(file_path, audio_data, sr)
    return file_path