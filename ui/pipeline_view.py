import tkinter as tk

class PipelineView:
    def __init__(self, parent, orchestrator, log_callback):
        self.parent = parent
        self.orchestrator = orchestrator
        self.log_callback = log_callback
        
        # State display
        self.state_label = tk.Label(self.parent, text=f"State: {self.orchestrator.state}", 
                                    font=("Arial", 14, "bold"), fg="red", bg="#f0f0f0")
        self.state_label.pack(pady=10)
        self.listen_btn = tk.Button(self.parent, text="🎙️ SPEAK", font=("Arial", 14, "bold"), 
                                    bg="lightblue", command=self.start_pipeline_thread)
        self.listen_btn.pack(pady=10, fill="x", padx=20)

        # The 7 Core Modules
        self.modules = [
            (1, "1. User Verification"),
            (2, "2. Wake Word Detection"),
            (3, "3. ASR (Whisper)"),
            (4, "4. Intent Detection"),
            (5, "5. Fulfillment"),
            (6, "6. Answer Generation"),
            (7, "7. Text-To-Speech")
        ]

        self.buttons = {}
        tk.Label(self.parent, text="Pipeline Bypasses", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=5)

        # Create a frame and bypass button for each module
        for step_num, name in self.modules:
            frame = tk.Frame(self.parent, bg="#f0f0f0")
            frame.pack(fill="x", padx=10, pady=5)
            
            lbl = tk.Label(frame, text=name, width=20, anchor="w", bg="#f0f0f0")
            lbl.pack(side="left")
            
            btn = tk.Button(frame, text="Bypass", command=lambda n=step_num: self.handle_bypass(n))
            btn.pack(side="right")
            self.buttons[step_num] = btn
        
        self.orchestrator.state_callback = self.update_state_label

    def handle_bypass(self, step_num):
        # Call the orchestrator logic
        result = self.orchestrator.bypass_step(step_num)
        
        # Update State Label if step 1 or 2 was bypassed
        # if step_num in [1, 2]:
        #     self.state_label.config(text=f"State: {result}")
        #     if result == "Awake":
        #         self.state_label.config(fg="green")
        #     else:
        #         self.state_label.config(fg="orange")
        
        # Log to the game view
        self.log_callback(f"Bypassed Step {step_num}: {result}")

    def start_pipeline_thread(self):
        """Runs the pipeline in a separate thread to prevent freezing the GUI."""
        import threading
        threading.Thread(target=self.orchestrator.run_full_pipeline, daemon=True).start()

    def update_state_label(self, new_state):
        color = {"Locked": "red", "Sleep": "orange", "Awake": "green"}.get(new_state, "black")
        self.state_label.config(text=f"State: {new_state}", fg=color)