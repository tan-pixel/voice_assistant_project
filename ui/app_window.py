import tkinter as tk
from tkinter import ttk
from ui.pipeline_view import PipelineView
from ui.game_view import GameView
import threading

class VoiceAssistantUI:
    def __init__(self, root, orchestrator):
        self.root = root
        self.orchestrator = orchestrator
        self.root.title("Atlas Voice Assistant - Group 15")
        self.root.geometry("1000x600")
        
        # Configure grid for the main window (2 columns)
        self.root.columnconfigure(0, weight=1) # Pipeline gets 1 part width
        self.root.columnconfigure(1, weight=2) # Game view gets 2 parts width
        self.root.rowconfigure(0, weight=1)

        # Build Left Panel: Pipeline & Bypasses
        self.pipeline_frame = tk.Frame(self.root, bg="#f0f0f0", bd=2, relief="groove")
        self.pipeline_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.pipeline_view = PipelineView(self.pipeline_frame, self.orchestrator, self.update_game_log)

        self.build_bypass_panel(self.pipeline_frame)

        # Build Right Panel: Dungeon Crawler System
        self.game_frame = tk.Frame(self.root, bd=2, relief="groove")
        self.game_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.game_view = GameView(self.game_frame)

    def build_bypass_panel(self, parent_frame):
        """Builds the left panel with inputs to satisfy the project bypass requirements."""
        tk.Label(parent_frame, text="Pipeline Bypasses", font=("Arial", 12, "bold")).pack(pady=10)

        # Step 1: User Verification (Passcode)
        tk.Label(parent_frame, text="1. Verification (Code):").pack(anchor="w", padx=5)
        f1 = tk.Frame(parent_frame)
        f1.pack(fill="x", padx=5, pady=2)
        self.code_input = tk.Entry(f1, width=15)
        self.code_input.pack(side="left", padx=2)
        tk.Button(f1, text="Unlock", command=lambda: self.trigger_bypass(1, self.code_input.get())).pack(side="right")

        # Step 2: Wake Word (Text Input)
        tk.Label(parent_frame, text="2. Wake Word (Type):").pack(anchor="w", padx=5, pady=(10,0))
        f2 = tk.Frame(parent_frame)
        f2.pack(fill="x", padx=5, pady=2)
        self.wake_input = tk.Entry(f2, width=15)
        self.wake_input.pack(side="left", padx=2)
        tk.Button(f2, text="Wake", command=lambda: self.trigger_bypass(2, self.wake_input.get())).pack(side="right")

        # Step 3: ASR (Text Injection)
        tk.Label(parent_frame, text="3. ASR (Type command):").pack(anchor="w", padx=5, pady=(10,0))
        f3 = tk.Frame(parent_frame)
        f3.pack(fill="x", padx=5, pady=2)
        self.asr_input = tk.Entry(f3, width=15)
        self.asr_input.pack(side="left", padx=2)
        tk.Button(f3, text="Send", command=lambda: self.trigger_bypass(3, self.asr_input.get())).pack(side="right")

        # Step 4: Intent/Slots (Dropdown + Text)
        tk.Label(parent_frame, text="4. Intent & Slots:").pack(anchor="w", padx=5, pady=(10,0))
        self.intent_combo = ttk.Combobox(parent_frame, values=["Weather", "Timer", "move", "lookup_monster", "Greetings"], state="readonly")
        self.intent_combo.set("Weather")
        self.intent_combo.pack(fill="x", padx=5, pady=2)
        self.intent_combo.bind("<<ComboboxSelected>>", self._on_intent_changed)
        f4 = tk.Frame(parent_frame)
        f4.pack(fill="x", padx=5, pady=2)
        self.slot_input = tk.Entry(f4, width=15)
        self.slot_input.insert(0, "city: Paris") # Placeholder example
        self.slot_input.pack(side="left", padx=2)
        tk.Button(f4, text="Inject", command=self._inject_intent_bypass).pack(side="right")

        # Step 5 & 6: Pre-Canned Buttons
        tk.Label(parent_frame, text="5 & 6. Pre-Canned:").pack(anchor="w", padx=5, pady=(15,0))
        tk.Button(parent_frame, text="Bypass 5: Canned API", command=lambda: self.trigger_bypass(5, None)).pack(fill="x", padx=5, pady=2)
        tk.Button(parent_frame, text="Bypass 6: Canned NL", command=lambda: self.trigger_bypass(6, None)).pack(fill="x", padx=5, pady=2)

    def _on_intent_changed(self, event):
        """Automatically fills the slot input with a default template when the dropdown changes."""
        selected_intent = self.intent_combo.get()
        
        # Clear whatever is currently in the text box
        self.slot_input.delete(0, tk.END)
        
        # Insert the correct pre-written template
        if selected_intent == "Weather":
            self.slot_input.insert(0, "city: Paris")
        elif selected_intent == "Timer":
            self.slot_input.insert(0, "duration: 5 minutes")
        elif selected_intent == "move":
            self.slot_input.insert(0, "direction: north")
        elif selected_intent == "lookup_monster":
            self.slot_input.insert(0, "name: badger")
        elif selected_intent == "Greetings":
            self.slot_input.insert(0, "") # Greetings don't require slots

    def _inject_intent_bypass(self):
        """Helper to package the intent and slots into a dictionary before sending."""
        intent = self.intent_combo.get()
        slot_text = self.slot_input.get()
        # Simple parser: "city: Paris" -> {"city": "Paris"}
        slots = {}
        if ":" in slot_text:
            k, v = slot_text.split(":", 1)
            slots[k.strip()] = v.strip()
            
        data = {"intent": intent, "slots": slots}
        self.trigger_bypass(4, data)

    def trigger_bypass(self, step, data):
        """Runs the bypass in a background thread to prevent UI freezing."""
        threading.Thread(target=self.orchestrator.manual_override, args=(step, data), daemon=True).start()

    def update_game_log(self, text):
        """Allows the pipeline to send messages to the game UI log."""
        self.game_view.update_log(text)