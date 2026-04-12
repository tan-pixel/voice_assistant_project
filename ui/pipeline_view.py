import tkinter as tk
import threading

class PipelineView:
    def __init__(self, parent, orchestrator, log_callback):
        self.parent = parent
        self.orchestrator = orchestrator
        self.log_callback = log_callback
        
        # State display
        self.state_label = tk.Label(self.parent, text=f"State: {self.orchestrator.state}", 
                                    font=("Arial", 14, "bold"), fg="red", bg="#f0f0f0")
        self.state_label.pack(pady=10)
        
        # Main Voice Command Button
        self.listen_btn = tk.Button(self.parent, text="🎙️ SPEAK", font=("Arial", 14, "bold"), 
                                    bg="lightblue", command=self.start_pipeline_thread)
        self.listen_btn.pack(pady=10, fill="x", padx=20)

        # Link the state callback so the label updates automatically
        self.orchestrator.state_callback = self.update_state_label

    def start_pipeline_thread(self):
        """Runs the pipeline in a separate thread to prevent freezing the GUI."""
        threading.Thread(target=self.orchestrator.run_full_pipeline, daemon=True).start()

    def update_state_label(self, new_state):
        color = {"Locked": "red", "Sleep": "orange", "Awake": "green"}.get(new_state, "black")
        self.state_label.config(text=f"State: {new_state}", fg=color)