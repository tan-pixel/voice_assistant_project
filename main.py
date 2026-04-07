import tkinter as tk
import threading
from ui.app_window import VoiceAssistantUI
from utils.audio_utils import record_audio

from core.m1_verification import UserVerificationModule
from core.m2_wake_word import WakeWordDetectionModule
from core.m3_asr import ASRModule
from core.m4_intent import IntentDetectionModule
from core.m5_fulfillment import FulfillmentModule
from core.m6_generation import AnswerGenerationModule
from core.m7_tts import TTSModule

class PipelineOrchestrator:
    def __init__(self):
        self.state = "Locked"
        self.ui_callback = None 
        self.dashboard_callback = None 
        self.game_update_callback = None
        
        # Initialize modules
        print("Initializing Core Modules...")
        self.m1 = UserVerificationModule()
        self.m2 = WakeWordDetectionModule()
        self.m3 = ASRModule()
        self.m4 = IntentDetectionModule()
        self.m5 = FulfillmentModule()
        self.m6 = AnswerGenerationModule()
        self.m7 = TTSModule()
        print("All modules loaded and ready.")

    def log(self, message):
        if self.ui_callback:
            self.ui_callback(message)
        print(message)

    def run_full_pipeline(self):
        """Executes the 7 steps in order, respecting the current system state."""
        audio_path = "data/live_input.wav"
        
        self.log("\n*** Starting Pipeline ***")
        
        # Record Audio
        self.log("Recording audio...")
        record_audio(audio_path, duration=3.0)

        # User Verification
        if self.state == "Locked":
            self.log("Step 1: Running Verification...")
            self.state = self.m1.process(audio_path)
            self.log(f"System State: {self.state}")
            if self.state == "Locked": 
                self.log("Verification failed. Try again or Bypass.")
                return
        else:
            self.log("Step 1: Bypassed (System is already Unlocked)")

        # Wake Word
        if self.state == "Sleep":
            self.log("Step 2: Running Wake Word Detection...")
            self.state = self.m2.process(audio_path)
            self.log(f"System State: {self.state}")
            if self.state == "Sleep": 
                self.log("Wake word not detected. Try again or Bypass.")
                return
        else:
            self.log("Step 2: Bypassed (System is already Awake)")

        # ASR
        self.log("Step 3: Running ASR (Whisper)...")
        text = self.m3.process(audio_path)
        self.log(f"Transcribed: '{text}'")
        if not text: return

        # Intent Detection
        self.log("Step 4: Running Intent Detection...")
        intent_data = self.m4.process(text)
        self.log(f"Detected Intent: {intent_data}")

        # Fulfillment
        self.log("Step 5: Running Fulfillment...")
        fulfillment_result = self.m5.process(intent_data)
        source = fulfillment_result.get("source")
        self.log(f"Fulfillment output: {source}")

        # Answer Generation
        self.log("Step 6: Generating Response...")
        nl_response = self.m6.process(fulfillment_result)
        self.log(f"Response: {nl_response}")

        # Update the UI Dashboard if it's an API call
        if source in ["weather_api", "timer", "dnd_api"]:
            if self.dashboard_callback:
                self.dashboard_callback(source, nl_response)

        elif source == "game_engine":
            if self.game_update_callback:
                game_state = fulfillment_result.get("data", {}).get("state", {})
                self.game_update_callback(game_state)

        # TTS
        self.log("Step 7: Playing TTS...")
        self.m7.process(nl_response)
        
        # Reset state back to Sleep after a successful command execution
        # We can comment this out if we want the system to stay awake forever after unlocking (maybe adjust later)
        self.state = "Sleep"
        self.log("System returning to Sleep mode.")
        self.log("*** Pipeline Complete ***\n")

    def bypass_step(self, step_num):
        """Updates the system state and the UI."""
        self.log(f"*** Bypassing Step {step_num} ***")
        if step_num == 1:
            self.state = "Sleep"
            return "Sleep"
        elif step_num == 2:
            self.state = "Awake"
            return "Awake"
            
        return f"Bypassed {step_num}"

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    root = tk.Tk()
    app = VoiceAssistantUI(root, orchestrator)
    
    # Wire up the callbacks so the orchestrator can talk to the UI safely
    orchestrator.ui_callback = app.update_game_log
    orchestrator.dashboard_callback = app.game_view.update_dashboard
    orchestrator.game_update_callback = app.game_view.update_game_visuals
    
    root.mainloop()