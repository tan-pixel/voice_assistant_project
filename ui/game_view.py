import tkinter as tk

class GameView:
    def __init__(self, parent):
        self.parent = parent
        
        # Configure rows: Top for APIs/Timer (Weight 1), Bottom for Game (Weight 2)
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=2)
        self.parent.columnconfigure(0, weight=1)

        # *** TOP PANEL: API Dashboard ***
        self.dashboard_frame = tk.Frame(self.parent, bg="#e0e0e0", bd=2, relief="ridge")
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        tk.Label(self.dashboard_frame, text="API Dashboard", font=("Arial", 12, "bold"), bg="#e0e0e0").pack(pady=2)
        
        self.weather_label = tk.Label(self.dashboard_frame, text="🌤️ Weather: Waiting...", font=("Arial", 10), bg="white", relief="solid")
        self.weather_label.pack(side="left", padx=10, fill="both", expand=True)
        
        self.timer_label = tk.Label(self.dashboard_frame, text="⏱️ Timer: 00:00", font=("Arial", 14, "bold"), bg="black", fg="white")
        self.timer_label.pack(side="left", padx=10, fill="both", expand=True)
        
        self.dnd_label = tk.Label(self.dashboard_frame, text="🐉 D&D Info: Waiting...", font=("Arial", 10), bg="white", relief="solid")
        self.dnd_label.pack(side="left", padx=10, fill="both", expand=True)

        # *** BOTTOM PANEL: Dungeon Crawler ***
        self.game_container = tk.Frame(self.parent)
        self.game_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.game_container.columnconfigure(0, weight=3) # Map
        self.game_container.columnconfigure(1, weight=1) # Stats
        self.game_container.rowconfigure(0, weight=1)

        # Left: Map Canvas
        self.canvas = tk.Canvas(self.game_container, bg="#eef2e6") 
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Draw Initial Placeholder
        self.canvas.create_text(250, 200, text="Say 'Restart' or 'Move' to load map", font=("Arial", 14), fill="gray")

        # Right: Stats Panel
        self.stats_frame = tk.Frame(self.game_container, bg="#d4e157")
        self.stats_frame.grid(row=0, column=1, sticky="nsew")
        
        # Stats Labels (Saved as variables so we can update them)
        self.hp_label = tk.Label(self.stats_frame, text="HP: --/--", font=("Arial", 12, "bold"), bg="#d4e157")
        self.hp_label.pack(pady=(15, 5))
        
        self.weapon_label = tk.Label(self.stats_frame, text="Weapon: --", font=("Arial", 12), bg="#d4e157", relief="ridge", width=15)
        self.weapon_label.pack(pady=5)
        
        self.pos_label = tk.Label(self.stats_frame, text="Position: [-, -]", font=("Arial", 12), bg="#d4e157", relief="ridge", width=15)
        self.pos_label.pack(pady=5)
        
        tk.Label(self.stats_frame, text="Inventory", font=("Arial", 12, "bold"), bg="#d4e157").pack(pady=(10, 0))
        self.inventory = tk.Listbox(self.stats_frame, height=5, font=("Arial", 10))
        self.inventory.pack(pady=5, padx=10, fill="x")

        tk.Label(self.stats_frame, text="Action Log", font=("Arial", 12, "bold"), bg="#d4e157").pack(pady=(10, 0))
        self.log_text = tk.Text(self.stats_frame, height=8, width=20, font=("Arial", 10), state="disabled")
        self.log_text.pack(pady=5, padx=10, fill="both", expand=True)

    def update_log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        
    def update_dashboard(self, source, text):
        if source == "weather_api":
            self.weather_label.config(text=f"🌤️ {text}")
        elif source == "timer":
            self.timer_label.config(text=f"⏱️ {text}")
        elif source == "dnd_api":
            short_text = (text[:40] + '...') if len(text) > 40 else text
            self.dnd_label.config(text=f"🐉 {short_text}")

    def update_game_visuals(self, state):
        """Redraws the map and updates the stats panel based on live game state."""
        # Update Stats Panel
        self.hp_label.config(text=f"HP: {state['hp']}/{state['max_hp']}")
        self.weapon_label.config(text=f"Weapon: {state['weapon']}")
        self.pos_label.config(text=f"Position: {state['position']}")
        
        self.inventory.delete(0, tk.END)
        for i, item in enumerate(state['inventory'], 1):
            self.inventory.insert(tk.END, f"{i}. {item}")

        # Draw the Map (Grid System)
        self.canvas.delete("all")
        cell_size = 50
        
        # Draw Map Items
        for pos_str, item_name in state['map_items'].items():
            y, x = map(int, pos_str.split(','))
            cx, cy = x * cell_size + 25, y * cell_size + 25
            
            # Simple visual representation of items
            icon = "🎁" if "Treasure" in item_name else "🧪" if "Potion" in item_name else "👹"
            self.canvas.create_text(cx, cy, text=icon, font=("Arial", 20))
            
        # Draw Player
        py, px = state['position']
        cx, cy = px * cell_size + 25, py * cell_size + 25
        self.canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill="lightblue", outline="blue", width=2)
        self.canvas.create_text(cx, cy, text="🧍", font=("Arial", 16))