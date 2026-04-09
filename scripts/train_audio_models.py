import os
import tensorflow as tf
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import random

# Configurations
DATA_DIR = "../data/raw_audio"
MODEL_DIR = "../data/models"
EVAL_DIR = "../data/evaluation"
TEAM_NAMES = ["nandal", "kubde", "khodabakhsh"]

TARGET_SR = 16000
DURATION = 2.0
NUM_SAMPLES = int(TARGET_SR * DURATION)
N_MFCC = 40
WINDOW_SEC = 0.025
HOP_SEC = 0.010

# Feature Extraction from Signal
def extract_mfcc_from_signal(y, sr):
    n_fft = int(WINDOW_SEC * sr)
    hop_length = int(HOP_SEC * sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc

# Augmentation
def augment_audio(y, sr):
    shift_max = int(0.2 * sr)
    shift = np.random.randint(-shift_max, shift_max)
    if shift > 0:
        y = np.pad(y, (shift, 0), mode='constant')[:-shift]
    elif shift < 0:
        y = np.pad(y, (0, -shift), mode='constant')[-shift:]
    n_steps = np.random.uniform(-2, 2)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y

# Collect File Paths and Labels
def collect_file_labels():
    """Returns lists of file paths and corresponding labels for both tasks."""
    file_paths = []
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
            file_paths.append(file_path)

            # Wake word label
            y_wakeword.append(1 if folder == "positive" else 0)

            # User verification label
            is_team = any(name in filename.lower() for name in TEAM_NAMES)
            y_verification.append(1 if is_team else 0)

    return file_paths, y_wakeword, y_verification

# Extract Features from a List of Files
def extract_features_from_files(file_list, augment=False, num_aug_copies=2):
    """Loads and processes audio files. If augment=True and file is positive,
    also creates augmented copies (only for wake word positive samples)."""
    X = []
    y_ww = []
    for fp in file_list:
        y, sr = librosa.load(fp, sr=TARGET_SR, mono=True)
        # Standardize length
        if len(y) < NUM_SAMPLES:
            y = np.pad(y, (0, NUM_SAMPLES - len(y)))
        else:
            y = y[:NUM_SAMPLES]

        # Original
        mfcc = extract_mfcc_from_signal(y, sr)
        X.append(mfcc)

        # Determine if positive (wake word) from path
        is_positive = "positive" in fp
        y_ww.append(1 if is_positive else 0)

        if augment and is_positive:
            for _ in range(num_aug_copies):
                y_aug = augment_audio(y.copy(), sr)
                mfcc_aug = extract_mfcc_from_signal(y_aug, sr)
                X.append(mfcc_aug)
                y_ww.append(1)   # still positive

    return np.array(X)[..., np.newaxis], np.array(y_ww)

# Model for User Verification (M1)
def build_cnn_m1(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Model for Wake Word (M2)
def build_cnn_m2(input_shape):
    """Smaller, regularized model for wake word detection."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # First conv block with L2 regularization
        layers.Conv2D(8, (3, 3), activation="relu",
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second conv block
        layers.Conv2D(16, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Global pooling instead of Flatten to reduce parameters
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),  # Focal loss
                  metrics=["accuracy"])
    return model

# Class Weights
def get_class_weights(y_train):
    total = len(y_train)
    pos = sum(y_train)
    neg = total - pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    return {0: weight_for_0, 1: weight_for_1}

# Evaluation Artifacts
def save_evaluation_artifacts(history, model, X_test, y_test, model_name):
    print(f"\nGenerating evaluation artifacts for {model_name}...")

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

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    report = classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"])
    with open(os.path.join(EVAL_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(os.path.join(EVAL_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    print("Collecting file paths and labels...")
    file_paths, y_ww_all, y_uv_all = collect_file_labels()
    print(f"Total audio files found: {len(file_paths)}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # *** M1: User Verification ***
    print("\n*** Training User Verification Model (M1) ***")
    X_clean, _ = extract_features_from_files(file_paths, augment=False)
    y_uv = np.array(y_uv_all)

    assert len(X_clean) == len(y_uv), \
        f"Feature/label count mismatch: {len(X_clean)} vs {len(y_uv)}"

    X_train_uv, X_test_uv, y_train_uv, y_test_uv = train_test_split(
        X_clean, y_uv, test_size=0.3, random_state=42, stratify=y_uv
    )
    uv_model = build_cnn_m1(X_clean.shape[1:])
    uv_class_weight = get_class_weights(y_train_uv)
    early_stop_uv = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    uv_history = uv_model.fit(
        X_train_uv, y_train_uv,
        epochs=100, batch_size=16, validation_split=0.2,
        class_weight=uv_class_weight, callbacks=[early_stop_uv], verbose=1
    )
    uv_loss, uv_acc = uv_model.evaluate(X_test_uv, y_test_uv, verbose=0)
    print(f"User Verification Test Accuracy: {uv_acc:.2f}")
    uv_model.save(os.path.join(MODEL_DIR, "verification_model.keras"))
    save_evaluation_artifacts(uv_history, uv_model, X_test_uv, y_test_uv, "M1_Verification")

    # *** M2: Wake Word ***
    print("\n*** Training Wake Word Model (M2) ***")
    train_files, test_files, _, y_ww_test_check = train_test_split(
        file_paths, y_ww_all, test_size=0.3, random_state=42, stratify=y_ww_all
    )

    X_train_ww, y_train_ww = extract_features_from_files(train_files, augment=True, num_aug_copies=2)
    X_test_ww, y_test_ww = extract_features_from_files(test_files, augment=False)

    # Verify test labels match expected (cross-check with split labels)
    assert np.array_equal(y_test_ww, np.array(y_ww_test_check)), \
        "Test labels from extract_features_from_files don't match split labels!"

    # Shuffle augmented training data so validation_split gets a representative slice
    shuffle_idx = np.random.permutation(len(X_train_ww))
    X_train_ww = X_train_ww[shuffle_idx]
    y_train_ww = y_train_ww[shuffle_idx]

    print(f"Training set size (after augmentation): {len(X_train_ww)}")
    print(f"Test set size (clean, no augmentation): {len(X_test_ww)}")
    print(f"Training positive rate: {y_train_ww.mean():.2%}")
    print(f"Test positive rate:     {y_test_ww.mean():.2%}")

    ww_model = build_cnn_m2(X_train_ww.shape[1:])
    ww_class_weight = get_class_weights(y_train_ww)
    early_stop_ww = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    ww_history = ww_model.fit(
        X_train_ww, y_train_ww,
        epochs=150, batch_size=16, validation_split=0.2,
        class_weight=ww_class_weight, callbacks=[early_stop_ww], verbose=1
    )
    ww_loss, ww_acc = ww_model.evaluate(X_test_ww, y_test_ww, verbose=0)
    print(f"Wake Word Test Accuracy: {ww_acc:.2f}")
    ww_model.save(os.path.join(MODEL_DIR, "wakeword_model.keras"))
    save_evaluation_artifacts(ww_history, ww_model, X_test_ww, y_test_ww, "M2_WakeWord")

    print(f"\nAll models trained and saved. Evaluation artifacts in {EVAL_DIR}")