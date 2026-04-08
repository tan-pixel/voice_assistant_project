import os
import tensorflow as tf
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configurations
DATA_DIR = "../data/raw_audio" 
MODEL_DIR = "../data/models"
EVAL_DIR = "../data/evaluation"

# team members' last names
TEAM_NAMES = ["nandal", "kubde", "khodabakhsh"] 

# Audio processing constants
TARGET_SR = 16000
DURATION = 2.0
NUM_SAMPLES = int(TARGET_SR * DURATION)
N_MFCC = 13
WINDOW_SEC = 0.025
HOP_SEC = 0.025

def extract_mfcc(file_path):
    """Loads audio, standardizes length, and extracts MFCCs."""
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)))
    else:
        y = y[:NUM_SAMPLES]
        
    n_fft = int(WINDOW_SEC * sr)
    hop_length = int(HOP_SEC * sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop_length)
    return mfcc

def load_and_label_data():
    """Reads the dataset and generates separate labels for M1 and M2."""
    X = []
    y_wakeword = []
    y_verification = []
    
    folders = ["positive", "near", "other"]
    
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} not found.")
            continue
            
        for filename in os.listdir(folder_path):
            if not filename.endswith((".wav", ".m4a", ".mp3")):
                continue
                
            file_path = os.path.join(folder_path, filename)
            mfcc = extract_mfcc(file_path)
            X.append(mfcc)
            
            # M2: Wake Word Labeling
            # 1 if it's the wake word ("positive" folder), 0 otherwise
            if folder == "positive":
                y_wakeword.append(1)
            else:
                y_wakeword.append(0)
                
            # M1: User Verification Labeling
            # 1 if the filename contains a team member's name, 0 otherwise
            is_team_member = any(name in filename.lower() for name in TEAM_NAMES)
            if is_team_member:
                y_verification.append(1)
            else:
                y_verification.append(0)

    # Convert to numpy arrays and add channel dimension for the CNN
    X = np.array(X)[..., np.newaxis]
    y_wakeword = np.array(y_wakeword)
    y_verification = np.array(y_verification)
    
    return X, y_wakeword, y_verification

def build_cnn(input_shape):
    """Builds the CNN architecture"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def save_evaluation_artifacts(history, model, X_test, y_test, model_name):
    """Generates and saves learning curves, confusion matrix, and metrics."""
    print(f"\nGenerating evaluation artifacts for {model_name}...")
    
    # Plot Training vs Validation Curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"{model_name}_learning_curves.png"))
    plt.close()

    # Generate Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Save Classification Report (Precision, Recall, F1)
    report = classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"])
    with open(os.path.join(EVAL_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)
    print(report)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(os.path.join(EVAL_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Extracting audio features. This may take a moment...")
    X, y_wakeword, y_verification = load_and_label_data()
    
    if len(X) == 0:
        raise ValueError("No audio data found. Please check your DATA_DIR path.")
        
    print(f"Extracted features for {len(X)} audio files. Input shape: {X.shape}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Train Wake Word Model (M2)
    print("\n*** Training Wake Word Model (M2) ***")
    X_train_ww, X_test_ww, y_train_ww, y_test_ww = train_test_split(
        X, y_wakeword, test_size=0.3, random_state=42, stratify=y_wakeword
    )
    
    ww_model = build_cnn(X.shape[1:])
    ww_history = ww_model.fit(X_train_ww, y_train_ww, epochs=50, batch_size=4, validation_split=0.2, verbose=1)
    
    ww_loss, ww_acc = ww_model.evaluate(X_test_ww, y_test_ww, verbose=0)
    print(f"Wake Word Model Test Accuracy: {ww_acc:.2f}")
    
    ww_model_path = os.path.join(MODEL_DIR, "wakeword_model.h5")
    ww_model.save(ww_model_path)
    print(f"Saved Wake Word model to {ww_model_path}")

    save_evaluation_artifacts(ww_history, ww_model, X_test_ww, y_test_ww, "M2_WakeWord")
    
    # Train User Verification Model (M1)
    print("\n*** Training User Verification Model (M1) ***")
    X_train_uv, X_test_uv, y_train_uv, y_test_uv = train_test_split(
        X, y_verification, test_size=0.3, random_state=42, stratify=y_verification
    )
    
    uv_model = build_cnn(X.shape[1:])
    # Handling potential class imbalance for verification (36 positive vs 300+ negative) using class weights. 
    weight_for_0 = 1.0 
    weight_for_1 = (len(y_train_uv) - sum(y_train_uv)) / (sum(y_train_uv) + 1)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    uv_history = uv_model.fit(X_train_uv, y_train_uv, epochs=50, batch_size=4, validation_split=0.2, class_weight=class_weight, verbose=1)
    
    uv_loss, uv_acc = uv_model.evaluate(X_test_uv, y_test_uv, verbose=0)
    print(f"User Verification Model Test Accuracy: {uv_acc:.2f}")
    
    uv_model_path = os.path.join(MODEL_DIR, "verification_model.h5")
    uv_model.save(uv_model_path)
    print(f"Saved User Verification model to {uv_model_path}")

    save_evaluation_artifacts(uv_history, uv_model, X_test_uv, y_test_uv, "M1_Verification")
    
    print(f"\nAll audio models trained and saved successfully! Evaluation graphs saved in {EVAL_DIR}")