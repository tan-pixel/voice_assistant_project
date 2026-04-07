import tkinter as tk
from ui.pipeline_view import PipelineView
from ui.game_view import GameView

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

        # Build Right Panel: Dungeon Crawler System
        self.game_frame = tk.Frame(self.root, bd=2, relief="groove")
        self.game_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.game_view = GameView(self.game_frame)

    def update_game_log(self, text):
        """Allows the pipeline to send messages to the game UI log."""
        self.game_view.update_log(text)