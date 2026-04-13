# Atlas Voice Assistant - Group 15

Developed for **CSI5180: Topics in AI Virtual Assistants** (Winter 2026).

## Team Members

- **Tanishk Nandal** (300447527)
- **Sharvari Kubde** (300143560)
- **Dorsa Khodabakhsh** (300444800)

## Project Overview

Atlas is an end-to-end, state-of-the-art voice assistant featuring a unified pipeline with GPU acceleration support. It integrates user authentication, natural language understanding, and a stateful control system.

### Core Pipeline Modules

| Module | Description |
|--------|-------------|
| **User Verification** | Text-independent binary classifier using MFCC features to authenticate team members. |
| **Wake Word Detection** | Real-time signature detection for the phrase "Hey Atlas". |
| **Automatic Speech Recognition (ASR)** | Powered by OpenAI Whisper. |
| **Intent Detection** | Joint Intent Classification and Slot Filling using a fine-tuned DistilBERT model. |
| **Fulfillment** | Orchestrates requests between the D&D 5e API, Open-Meteo Weather API, and an internal Game Engine. |
| **Answer Generation** | Transforms JSON data into natural language and updates the visual UI. |
| **Text-to-Speech (TTS)** | High-fidelity voice output via Edge-TTS. |

## Control System: Dungeon Crawler

Atlas controls a simulated 2D dungeon crawler game.

- **Stateful Environment**: Tracks HP, position, and inventory.
- **Visual Feedback**: Features a real-time grid map and an action log dashboard.
- **Commands**: Supports intents such as `move`, `pick_up`, and `use_item`.

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- FFmpeg (required for Whisper audio processing)
- NVIDIA GPU with CUDA support (strongly recommended for training and inference)

### 2. Install Dependencies

Creating a virtual environment is recommended before installing the dependencies.
```bash
pip install -r requirements.txt
pip install openwakeword onnxruntime
```

### 3. Training the Models
Before running the assistant, you must train the local neural networks:
```
cd scripts

# Train M1 (Verification)
python train_audio_models.py --m1-only

# Train M4 (BERT Intent Detection)
python train_intent_model.py
```

M2 now uses the `openWakeWord` framework with a custom `Hey Atlas` model.
Place that custom model at [data/models/openwakeword/README.md](/Users/dorsa/Desktop/uOttawa/Term 4/CSI5180/voice_assistant_project/data/models/openwakeword/README.md).

## Usage
Run the main orchestrator to launch the GUI:
```
python main.py
```
1. Click the **"SPEAK"** button.
2. Speak your command (e.g., *"Hey Atlas, move north"* or *"Hey Atlas, what are the stats for a Goblin?"*).
3. The assistant will verify your voice, transcribe the text, and fulfill the intent.

> **Bypasses**: Each step in the pipeline can be manually bypassed via the UI for testing.

## Project Structure

```
voice_assistant_project/
│
├── main.py                     # Orchestrates the pipeline and runs the main UI
├── requirements.txt            # Python dependencies
├── config.yaml                 # Global configurations (API endpoints, thresholds, paths)
│
├── core/                       # The 7 modules
│   ├── __init__.py
│   ├── m1_verification.py      # MFCC-based binary classifier for team authentication
│   ├── m2_wake_word.py         # openWakeWord-backed "Hey Atlas" detection
│   ├── m3_asr.py               # Whisper integration for speech-to-text
│   ├── m4_intent.py            # BERT-based intent detection and slot filling
│   ├── m5_fulfillment.py       # Routes intents to D&D API or Game Engine
│   ├── m6_generation.py        # LLM/Template natural language & visual state updates
│   └── m7_tts.py               # Text-to-speech engine
│
├── domain/                     # Specialized Domain (D&D 5e Knowledge)
│   ├── __init__.py
│   └── dnd_client.py           # Handles GET requests to https://www.dnd5eapi.co/
│
├── control_system/             # Simulated Dungeon Crawler
│   ├── __init__.py
│   ├── game_engine.py          # State management (HP, Inventory, Position)
│   └── map_renderer.py         # Logic for rendering the 2D grid and icons
│
├── utils/                      # Helper functions
│   ├── __init__.py
│   ├── audio_utils.py          # Audio recording, MFCC extraction, silence trimming
│   └── text_utils.py           # String cleaning, JSON parsing
│
├── data/                       # Local data storage (ignored in version control)
│   ├── raw_audio/              # Activity 1 dataset (team vs others)
│   ├── processed_features/     # Pickled/Numpy arrays of extracted MFCCs
│   └── models/                 # Saved weights (.pt/.h5) for classifiers and BERT
├── scripts/                       
│   ├── train_audio_models.py   # Train M1 (Verification); M2 uses openWakeWord
│   └── train_intent_model.py   # Train M4 (BERT Intent Detection)
│
└── ui/                         # User Interface components
    ├── __init__.py
    ├── app_window.py           # Main window layout
    ├── pipeline_view.py        # Displays the 7 steps and bypass controls
    └── game_view.py            # Renders the dungeon crawler map and stats panel
```

## APIs Used

- **D&D 5e API**: Specialized domain for fantasy knowledge.
- **Open-Meteo**: Weather intent fulfillment.
