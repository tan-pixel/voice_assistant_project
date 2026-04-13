import re
from datetime import datetime
import tkinter as tk

class GameView:
    DASHBOARD_RESET_MS = 9000

    def __init__(self, parent):
        self.parent = parent
        self.timer_after_id = None
        self.timer_seconds_remaining = 0
        self.timer_name = "Timer"
        self.alarm_after_id = None
        self.alarm_target = None
        self.alarm_name = "Alarm"
        self.alarm_display_time = "Waiting..."
        self.dashboard_reset_after_ids = {}
        
        # Configure rows: Top for APIs/Timers/Alarm (Weight 1), Bottom for Game (Weight 2)
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(1, weight=2)
        self.parent.columnconfigure(0, weight=1)

        # *** TOP PANEL: API Dashboard ***
        self.dashboard_frame = tk.Frame(self.parent, bg="#e0e0e0", bd=2, relief="ridge")
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.dashboard_frame.rowconfigure(1, weight=1)
        for column in range(5):
            self.dashboard_frame.columnconfigure(column, weight=1, uniform="dashboard")

        tk.Label(self.dashboard_frame, text="API Dashboard", font=("Arial", 12, "bold"), bg="#e0e0e0").grid(
            row=0, column=0, columnspan=5, pady=2
        )

        label_style = {
            "padx": 6,
            "pady": 4,
            "sticky": "nsew",
        }

        self.weather_label = tk.Label(
            self.dashboard_frame,
            text="🌤️ Weather: Waiting...",
            font=("Arial", 10),
            bg="white",
            relief="solid",
            anchor="w",
            justify="left",
            wraplength=170,
        )
        self.weather_label.grid(row=1, column=0, **label_style)

        self.timer_label = tk.Label(
            self.dashboard_frame,
            text="⏱️ Timer: 00:00",
            font=("Arial", 14, "bold"),
            bg="black",
            fg="white",
            relief="solid",
            anchor="w",
            justify="left",
            wraplength=170,
        )
        self.timer_label.grid(row=1, column=1, **label_style)

        self.alarm_label = tk.Label(
            self.dashboard_frame,
            text="⏰ Alarm: Waiting...",
            font=("Arial", 11, "bold"),
            bg="#fff8dc",
            relief="solid",
            anchor="w",
            justify="left",
            wraplength=170,
        )
        self.alarm_label.grid(row=1, column=2, **label_style)

        self.assistant_label = tk.Label(
            self.dashboard_frame,
            text="💡 Assistant: Waiting...",
            font=("Arial", 10),
            bg="white",
            relief="solid",
            anchor="w",
            justify="left",
            wraplength=170,
        )
        self.assistant_label.grid(row=1, column=3, **label_style)

        self.dnd_label = tk.Label(
            self.dashboard_frame,
            text="🐉 D&D Info: Waiting...",
            font=("Arial", 10),
            bg="white",
            relief="solid",
            anchor="w",
            justify="left",
            wraplength=170,
        )
        self.dnd_label.grid(row=1, column=4, **label_style)

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
        self.canvas.create_text(250, 200, text="Say 'restart game' to load the dungeon", font=("Arial", 14), fill="gray")

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
        
    def update_dashboard(self, source, text, payload=None):
        self.parent.after(0, self._apply_dashboard_update, source, text, payload)

    def _apply_dashboard_update(self, source, text, payload=None):
        if source == "weather_api":
            self.weather_label.config(text=f"🌤️ {text}")
            self._schedule_dashboard_reset("weather")
        elif source == "timer":
            self._start_timer(payload, text)
        elif source == "alarm":
            self._start_alarm(payload, text)
        elif source == "system":
            short_text = (text[:70] + '...') if len(text) > 70 else text
            self.assistant_label.config(text=f"💡 {short_text}")
            self._schedule_dashboard_reset("assistant")
        elif source == "dnd_api":
            short_text = (text[:40] + '...') if len(text) > 40 else text
            self.dnd_label.config(text=f"🐉 {short_text}")
            self._schedule_dashboard_reset("dnd")

    def _start_timer(self, payload, fallback_text):
        timer_data = (payload or {}).get("data", {})
        duration_text = timer_data.get("duration", "")
        timer_name = timer_data.get("name", "Timer") or "Timer"
        total_seconds = self._parse_duration_to_seconds(duration_text)

        if self.timer_after_id is not None:
            self.parent.after_cancel(self.timer_after_id)
            self.timer_after_id = None

        if total_seconds <= 0:
            self.timer_label.config(text=f"⏱️ {fallback_text}")
            return

        self.timer_seconds_remaining = total_seconds
        self.timer_name = timer_name
        self._render_timer_label()
        self.timer_after_id = self.parent.after(1000, self._tick_timer)

    def _start_alarm(self, payload, fallback_text):
        alarm_data = (payload or {}).get("data", {})
        target_iso = alarm_data.get("target_iso")
        alarm_name = alarm_data.get("name", "Alarm") or "Alarm"
        target_display = alarm_data.get("target_display")

        if self.alarm_after_id is not None:
            self.parent.after_cancel(self.alarm_after_id)
            self.alarm_after_id = None

        if not target_iso:
            self.alarm_display_time = "Waiting..."
            self.alarm_label.config(text=f"⏰ {fallback_text}")
            return

        try:
            self.alarm_target = datetime.fromisoformat(target_iso)
        except ValueError:
            self.alarm_target = None
            self.alarm_display_time = "Waiting..."
            self.alarm_label.config(text=f"⏰ {fallback_text}")
            return

        self.alarm_name = alarm_name
        self.alarm_display_time = target_display or self.alarm_target.strftime("%a %I:%M %p")
        self._render_alarm_label()
        self.alarm_after_id = self.parent.after(1000, self._tick_alarm)

    def _tick_timer(self):
        if self.timer_seconds_remaining > 0:
            self.timer_seconds_remaining -= 1
            self._render_timer_label()

        if self.timer_seconds_remaining > 0:
            self.timer_after_id = self.parent.after(1000, self._tick_timer)
        else:
            self.timer_after_id = None
            self.timer_label.config(text=f"⏱️ {self.timer_name}: Done")

    def _tick_alarm(self):
        if self.alarm_target is None:
            self.alarm_after_id = None
            return

        remaining = int((self.alarm_target - datetime.now()).total_seconds())
        if remaining > 0:
            self.alarm_after_id = self.parent.after(1000, self._tick_alarm)
        else:
            self.alarm_after_id = None
            self.alarm_label.config(text=f"⏰ {self.alarm_name}: Ringing")

    def _render_timer_label(self):
        self.timer_label.config(
            text=f"⏱️ {self.timer_name}: {self._format_duration(self.timer_seconds_remaining)}"
        )

    def _render_alarm_label(self):
        self.alarm_label.config(
            text=f"⏰ {self.alarm_name}: {self.alarm_display_time}"
        )

    def _schedule_dashboard_reset(self, panel_name):
        existing = self.dashboard_reset_after_ids.get(panel_name)
        if existing is not None:
            self.parent.after_cancel(existing)

        self.dashboard_reset_after_ids[panel_name] = self.parent.after(
            self.DASHBOARD_RESET_MS,
            lambda: self._reset_dashboard_panel(panel_name),
        )

    def _reset_dashboard_panel(self, panel_name):
        self.dashboard_reset_after_ids[panel_name] = None
        if panel_name == "weather":
            self.weather_label.config(text="🌤️ Weather: Waiting...")
        elif panel_name == "assistant":
            self.assistant_label.config(text="💡 Assistant: Waiting...")
        elif panel_name == "dnd":
            self.dnd_label.config(text="🐉 D&D Info: Waiting...")

    @staticmethod
    def _format_duration(total_seconds):
        hours, remainder = divmod(max(total_seconds, 0), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @classmethod
    def _parse_duration_to_seconds(cls, duration_text):
        if not duration_text:
            return 0

        text = duration_text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\bmins\b", "minutes", text)
        text = re.sub(r"\bmin\b", "minute", text)
        text = re.sub(r"\bhrs\b", "hours", text)
        text = re.sub(r"\bhr\b", "hour", text)
        text = re.sub(r"\s+", " ", text).strip()

        units = {
            "second": 1,
            "seconds": 1,
            "minute": 60,
            "minutes": 60,
            "hour": 3600,
            "hours": 3600,
        }

        total_seconds = 0
        quantity_tokens = []
        for token in text.split():
            if token in units:
                quantity = cls._parse_quantity(quantity_tokens)
                if quantity is None:
                    quantity = 1
                total_seconds += int(quantity * units[token])
                quantity_tokens = []
            else:
                quantity_tokens.append(token)

        return total_seconds

    @staticmethod
    def _parse_quantity(tokens):
        if not tokens:
            return None

        number_words = {
            "a": 1,
            "an": 1,
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }

        tokens = [token for token in tokens if token not in {"of"}]
        if not tokens:
            return None

        if "half" in tokens:
            if tokens == ["half"]:
                return 0.5
            prefix = [token for token in tokens if token != "half" and token != "and"]
            base = GameView._parse_simple_number(prefix) or 0
            return base + 0.5

        if "quarter" in tokens or "quarters" in tokens:
            quarter_tokens = [token for token in tokens if token not in {"a", "an", "and"}]
            if quarter_tokens == ["quarter"]:
                return 0.25
            if quarter_tokens == ["quarters"]:
                return 0.25
            if quarter_tokens[-1] in {"quarter", "quarters"}:
                base = GameView._parse_simple_number(quarter_tokens[:-1]) or 0
                if base == 0:
                    return 0.25
                return base * 0.25

        filtered = [token for token in tokens if token != "and"]
        return GameView._parse_simple_number(filtered)

    @staticmethod
    def _parse_simple_number(tokens):
        if not tokens:
            return None

        number_words = {
            "a": 1,
            "an": 1,
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }

        total = 0
        found = False
        for token in tokens:
            if token.isdigit():
                total += int(token)
                found = True
            elif token in number_words:
                total += number_words[token]
                found = True
        return total if found else None

    def update_game_visuals(self, state):
        """Redraws the map and updates the stats panel based on live game state."""
        if not isinstance(state, dict):
            return # Failsafe if state is completely broken
        
        # Update Stats Panel
        hp = state.get('hp', '--')
        max_hp = state.get('max_hp', '--')
        self.hp_label.config(text=f"HP: {hp}/{max_hp}")
        
        self.weapon_label.config(text=f"Weapon: {state.get('weapon', 'None')}")
        self.pos_label.config(text=f"Position: {state.get('position', ['-', '-'])}")
        
        self.inventory.delete(0, tk.END)
        for i, item in enumerate(state.get('inventory', []), 1):
            self.inventory.insert(tk.END, f"{i}. {item}")

        # Draw the Map (Grid System)
        self.canvas.delete("all")
        cell_size = 50
        
        # Draw Map Items
        for pos_str, item_name in state.get('map_items', {}).items():
            try:
                y, x = map(int, pos_str.split(','))
                cx, cy = x * cell_size + 25, y * cell_size + 25

                icon = self._icon_for_map_item(item_name)
                self.canvas.create_text(cx, cy, text=icon, font=("Arial", 20))
            except ValueError:
                pass # Skip if coordinate format is weird
            
        # Draw Player
        pos = state.get('position', [0, 0])
        if isinstance(pos, list) and len(pos) == 2:
            py, px = pos
            cx, cy = px * cell_size + 25, py * cell_size + 25
            self.canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill="lightblue", outline="blue", width=2)
            self.canvas.create_text(cx, cy, text="🧍", font=("Arial", 16))

        if state.get("victory"):
            self.canvas.create_text(200, 30, text="Treasure Opened", font=("Arial", 18, "bold"), fill="darkgreen")
        elif state.get("game_over"):
            self.canvas.create_text(200, 30, text="Game Over", font=("Arial", 18, "bold"), fill="darkred")

    @staticmethod
    def _icon_for_map_item(item_name):
        lowered = item_name.lower()
        if "opened treasure" in lowered:
            return "🏆"
        if "treasure" in lowered or "chest" in lowered:
            return "🎁"
        if "potion" in lowered:
            return "🧪"
        if "key" in lowered:
            return "🔑"
        if "sword" in lowered or "axe" in lowered or "weapon" in lowered:
            return "⚔️"
        if "goblin" in lowered or "monster" in lowered:
            return "👹"
        return "📦"
