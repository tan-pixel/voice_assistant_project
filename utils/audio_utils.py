import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

def extract_mfcc_for_inference(file_path, sr=16000, duration=2.0, n_mfcc=40, window_sec=0.025, hop_sec=0.010):
    """
    Loads an audio file, standardizes its length, and extracts MFCC features.
    Updated to match the N_MFCC=40, HOP_SEC=0.010, and standardization logic from training.
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
    
    # Standardize the MFCCs to match the training script
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    
    # Add batch and channel dimensions for Keras: shape becomes (1, mfcc_coeffs, time_steps, 1)
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc_input

def record_audio(file_path="data/live_input.wav", duration=3.0, sr=16000):
    """Records live audio from the microphone and saves it to a file."""
    print(f"Listening for {duration} seconds...")

    try:
        # Try to use the default input device
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        
        # Save as WAV file
        sf.write(file_path, audio_data, sr)
        return file_path
    except Exception as e:
        print(f"Error with default device: {e}")
        
        # Fallback: List available devices and try first input device
        print("Available devices:")
        devices = sd.query_devices()
        input_device = None
        
        for i, device in enumerate(devices):
            print(f"  {i}: {device}")
            if device['max_input_channels'] > 0 and input_device is None:
                input_device = i
        
        if input_device is None:
            raise RuntimeError("No input devices found. Please check your microphone.")
        
        print(f"Using input device: {input_device}")
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=input_device)
        sd.wait()
        print("Recording complete.")
        
        # Save as WAV file
        sf.write(file_path, audio_data, sr)
        return file_path