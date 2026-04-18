"""
game_view.py  –  Atlas Voice Assistant (Group 15)

Dark UI with green accents. Map canvas stays white/light so
colour emoji remain visible. Grid fills the full canvas and
redraws dynamically whenever the window is resized.
"""

import re
from datetime import datetime
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


# ── dark palette ──────────────────────────────────────────────────────────────
BG_CARD   = "#1e2822"
BG_PANEL  = "#263020"
BORDER    = "#3a5040"
ACCENT    = "#3dba72"
FG_HEAD   = "#e8f5ed"
FG_BODY   = "#b8d4c0"
FG_DIM    = "#5a7a62"
FG_GOOD   = "#3dba72"
FG_WARN   = "#e0a030"
FG_DANGER = "#e05555"
FG_GOLD   = "#d4a820"

# ── map palette (light so emoji are visible) ──────────────────────────────────
MAP_BG        = "#f8fdf9"
MAP_CELL_EVEN = "#f0f8f2"
MAP_CELL_ODD  = "#e8f4ec"
MAP_GRID      = "#c0dcc8"

FONT_H2   = ("Segoe UI", 10, "bold")
FONT_BODY = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


class GameView:
    DASHBOARD_RESET_MS = 9_000
    CELL = 50

    def __init__(self, parent):
        self.parent = parent

        # timer / alarm state
        self.timer_after_id            = None
        self.timer_seconds_remaining   = 0
        self.timer_name                = "Timer"
        self.alarm_after_id            = None
        self.alarm_target              = None
        self.alarm_name                = "Alarm"
        self.alarm_display_time        = "Waiting…"
        self.dashboard_reset_after_ids = {}

        # last known game state — needed so <Configure> can redraw correctly
        self._last_state = {}

        self._root = tk.Frame(parent, bg=BG_CARD)
        self._root.pack(fill="both", expand=True)
        self._root.rowconfigure(0, weight=1)
        self._root.rowconfigure(1, weight=2)
        self._root.columnconfigure(0, weight=1)

        self._build_dashboard(self._root)
        self._build_game_area(self._root)

    # ── dashboard ─────────────────────────────────────────────────────────────

    def _build_dashboard(self, parent):
        outer = tk.Frame(parent, bg=BG_CARD)
        outer.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        for c in range(5):
            outer.columnconfigure(c, weight=1, uniform="dash")
        outer.rowconfigure(1, weight=1)

        tk.Label(outer, text="📡  API Dashboard",
                 font=FONT_H2, fg=ACCENT, bg=BG_CARD, anchor="w"
                 ).grid(row=0, column=0, columnspan=5, sticky="w", padx=6, pady=(2, 6))

        def card(col, emoji, label_text, fg=FG_BODY):
            f = tk.Frame(outer, bg=BG_PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
            f.grid(row=1, column=col, sticky="nsew", padx=3, pady=2)
            f.rowconfigure(0, weight=1)
            f.columnconfigure(0, weight=1)
            lbl = tk.Label(f, text=f"{emoji}  {label_text}",
                           font=FONT_BODY, fg=fg, bg=BG_PANEL,
                           anchor="nw", justify="left",
                           wraplength=160, padx=8, pady=6)
            lbl.grid(sticky="nsew")
            return lbl

        self.weather_label   = card(0, "🌤️", "Weather: Waiting…")
        self.timer_label     = card(1, "⏱️", "Timer: 00:00",      fg=FG_GOOD)
        self.alarm_label     = card(2, "⏰", "Alarm: Waiting…",    fg=FG_WARN)
        self.assistant_label = card(3, "💡", "Assistant: Waiting…")
        self.dnd_label       = card(4, "🐉", "D&D Info: Waiting…")

    # ── game area ─────────────────────────────────────────────────────────────

    def _build_game_area(self, parent):
        area = tk.Frame(parent, bg=BG_CARD)
        area.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        area.columnconfigure(0, weight=3)
        area.columnconfigure(1, weight=1)
        area.rowconfigure(0, weight=1)

        # map canvas
        canvas_wrap = tk.Frame(area, bg=MAP_BG,
                               highlightbackground=BORDER, highlightthickness=1)
        canvas_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_wrap, bg=MAP_BG, highlightthickness=0)
        self.canvas.grid(sticky="nsew")

        # KEY FIX: redraw the full grid (and any game state) on every resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # stats panel
        stats = tk.Frame(area, bg=BG_PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        stats.grid(row=0, column=1, sticky="nsew")
        stats.columnconfigure(0, weight=1)

        def stat_lbl(text, fg=FG_HEAD, font=FONT_H2):
            lbl = tk.Label(stats, text=text, font=font,
                           fg=fg, bg=BG_PANEL, anchor="center", padx=6, pady=4)
            lbl.pack(fill="x", pady=3, padx=6)
            return lbl

        self.hp_label     = stat_lbl("❤️  HP: --/--",   fg=FG_DANGER)
        self.weapon_label = stat_lbl("⚔️  Weapon: --",  fg=FG_HEAD,  font=FONT_BODY)
        self.pos_label    = stat_lbl("📍  Pos: [-, -]", fg=FG_DIM,   font=FONT_BODY)

        tk.Frame(stats, bg=BORDER, height=1).pack(fill="x", padx=8, pady=4)

        tk.Label(stats, text="🎒  Inventory", font=FONT_H2,
                 fg=ACCENT, bg=BG_PANEL, anchor="w"
                 ).pack(fill="x", padx=8, pady=(4, 2))

        self.inventory = tk.Listbox(
            stats, height=5, font=FONT_MONO,
            bg=BG_CARD, fg=FG_BODY,
            selectbackground=ACCENT, selectforeground="#0a1a10",
            relief="flat", highlightthickness=1,
            highlightbackground=BORDER, bd=0,
        )
        self.inventory.pack(fill="x", padx=8, pady=(0, 4))

        tk.Frame(stats, bg=BORDER, height=1).pack(fill="x", padx=8, pady=4)

        tk.Label(stats, text="📜  Action Log", font=FONT_H2,
                 fg=ACCENT, bg=BG_PANEL, anchor="w"
                 ).pack(fill="x", padx=8, pady=(4, 2))

        self.log_text = tk.Text(
            stats, height=8, font=FONT_MONO,
            bg=BG_CARD, fg=FG_GOOD,
            insertbackground=FG_HEAD,
            relief="flat", highlightthickness=1,
            highlightbackground=BORDER, bd=0,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    # ── resize handler ────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        """Called automatically whenever the canvas changes size."""
        self._draw_map(self._last_state)

    # ── public API ────────────────────────────────────────────────────────────

    def update_log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def update_dashboard(self, source, text, payload=None):
        self.parent.after(0, self._apply_dashboard_update, source, text, payload)

    def _apply_dashboard_update(self, source, text, payload=None):
        if source == "weather_api":
            self.weather_label.config(text=f"🌤️  {text}")
            self._schedule_dashboard_reset("weather")
        elif source == "timer":
            self._start_timer(payload, text)
        elif source == "alarm":
            self._start_alarm(payload, text)
        elif source == "system":
            short = (text[:70] + "…") if len(text) > 70 else text
            self.assistant_label.config(text=f"💡  {short}")
            self._schedule_dashboard_reset("assistant")
        elif source == "dnd_api":
            short = (text[:40] + "…") if len(text) > 40 else text
            self.dnd_label.config(text=f"🐉  {short}")
            self._schedule_dashboard_reset("dnd")

    # ── timer / alarm ─────────────────────────────────────────────────────────

    def _start_timer(self, payload, fallback_text):
        data = (payload or {}).get("data", {})
        self.timer_name = data.get("name", "Timer") or "Timer"
        total = self._parse_duration_to_seconds(data.get("duration", ""))
        if self.timer_after_id:
            self.parent.after_cancel(self.timer_after_id)
            self.timer_after_id = None
        if total <= 0:
            self.timer_label.config(text=f"⏱️  {fallback_text}")
            return
        self.timer_seconds_remaining = total
        self._render_timer_label()
        self.timer_after_id = self.parent.after(1000, self._tick_timer)

    def _start_alarm(self, payload, fallback_text):
        data = (payload or {}).get("data", {})
        target_iso = data.get("target_iso")
        self.alarm_name = data.get("name", "Alarm") or "Alarm"
        target_display  = data.get("target_display")
        if self.alarm_after_id:
            self.parent.after_cancel(self.alarm_after_id)
            self.alarm_after_id = None
        if not target_iso:
            self.alarm_display_time = "Waiting…"
            self.alarm_label.config(text=f"⏰  {fallback_text}")
            return
        try:
            self.alarm_target = datetime.fromisoformat(target_iso)
        except ValueError:
            self.alarm_target = None
            self.alarm_display_time = "Waiting…"
            self.alarm_label.config(text=f"⏰  {fallback_text}")
            return
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
            self.timer_label.config(text=f"⏱️  {self.timer_name}: Done ✅", fg=FG_GOOD)

    def _tick_alarm(self):
        if self.alarm_target is None:
            self.alarm_after_id = None
            return
        remaining = int((self.alarm_target - datetime.now()).total_seconds())
        if remaining > 0:
            self.alarm_after_id = self.parent.after(1000, self._tick_alarm)
        else:
            self.alarm_after_id = None
            self.alarm_label.config(text=f"⏰  {self.alarm_name}: 🔔 Ringing!", fg=FG_DANGER)

    def _render_timer_label(self):
        self.timer_label.config(
            text=f"⏱️  {self.timer_name}:  {self._format_duration(self.timer_seconds_remaining)}"
        )

    def _render_alarm_label(self):
        self.alarm_label.config(text=f"⏰  {self.alarm_name}:  {self.alarm_display_time}")

    def _schedule_dashboard_reset(self, panel):
        existing = self.dashboard_reset_after_ids.get(panel)
        if existing:
            self.parent.after_cancel(existing)
        self.dashboard_reset_after_ids[panel] = self.parent.after(
            self.DASHBOARD_RESET_MS, lambda: self._reset_dashboard_panel(panel)
        )

    def _reset_dashboard_panel(self, panel):
        self.dashboard_reset_after_ids[panel] = None
        resets = {
            "weather":   ("🌤️  Weather: Waiting…",   self.weather_label,   FG_BODY),
            "assistant": ("💡  Assistant: Waiting…", self.assistant_label, FG_BODY),
            "dnd":       ("🐉  D&D Info: Waiting…",  self.dnd_label,       FG_BODY),
        }
        if panel in resets:
            text, lbl, fg = resets[panel]
            lbl.config(text=text, fg=fg)

    # ── map drawing ───────────────────────────────────────────────────────────

    def update_game_visuals(self, state):
        if not isinstance(state, dict):
            return

        self._last_state = state   # store so resize can redraw correctly

        hp, max_hp = state.get("hp", "--"), state.get("max_hp", "--")
        self.hp_label.config(text=f"❤️  HP: {hp}/{max_hp}")
        self.weapon_label.config(text=f"⚔️  {state.get('weapon', 'None')}")
        self.pos_label.config(text=f"📍  {state.get('position', ['-', '-'])}")

        self.inventory.delete(0, tk.END)
        for i, item in enumerate(state.get("inventory", []), 1):
            self.inventory.insert(tk.END, f"  {i}.  {item}")

        self._draw_map(state)

    def _draw_map(self, state):
        self.canvas.delete("all")
        cell = self.CELL

        # Use the real rendered size; both will be >1 once <Configure> fires
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w <= 1 or h <= 1:
            # Widget not yet laid out — show placeholder and wait for <Configure>
            self.canvas.create_text(
                150, 100,
                text="Say  'restart game'  to load the dungeon",
                font=("Segoe UI", 13), fill="#a0c0a8",
            )
            return

        # ── checkerboard cells filling the entire canvas ───────────────────
        cols = w // cell + 1
        rows = h // cell + 1
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * cell, r * cell
                fill = MAP_CELL_EVEN if (r + c) % 2 == 0 else MAP_CELL_ODD
                self.canvas.create_rectangle(
                    x0, y0, x0 + cell, y0 + cell,
                    fill=fill, outline=MAP_GRID, width=1,
                )

        # ── map items ─────────────────────────────────────────────────────
        for pos_str, item_name in state.get("map_items", {}).items():
            try:
                row, col = map(int, pos_str.split(","))
                cx = col * cell + cell // 2
                cy = row * cell + cell // 2
                self.canvas.create_text(cx, cy,
                                        text=self._icon_for_map_item(item_name),
                                        font=("Segoe UI Emoji", 22))
            except ValueError:
                pass

        # ── player ────────────────────────────────────────────────────────
        pos = state.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) == 2:
            py, px = pos
            cx = px * cell + cell // 2
            cy = py * cell + cell // 2
            r = 18
            self.canvas.create_rectangle(
                px * cell, py * cell,
                px * cell + cell, py * cell + cell,
                fill="#cceedd", outline=ACCENT, width=2,
            )
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill="#d8f5e8", outline=ACCENT, width=2)
            self.canvas.create_text(cx, cy, text="🧍",
                                    font=("Segoe UI Emoji", 16))

        # ── overlays ──────────────────────────────────────────────────────
        if state.get("victory"):
            self._overlay("🏆  Treasure Opened!", FG_GOLD, w)
        elif state.get("game_over"):
            self._overlay("💀  Game Over", FG_DANGER, w)

    def _overlay(self, text, color, w):
        self.canvas.create_rectangle(0, 0, w, 52, fill="#ffffffdd", outline="")
        self.canvas.create_text(w // 2, 26, text=text,
                                font=("Segoe UI", 15, "bold"), fill=color)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _icon_for_map_item(item_name):
        low = item_name.lower()
        if "opened treasure" in low:             return "🏆"
        if "treasure" in low or "chest" in low: return "🎁"
        if "potion"   in low:                   return "🧪"
        if "key"      in low:                   return "🔑"
        if "sword" in low or "axe" in low or "weapon" in low: return "⚔️"
        if "goblin" in low or "monster" in low: return "👹"
        return "📦"

    @staticmethod
    def _format_duration(total_seconds):
        h, rem = divmod(max(total_seconds, 0), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    @classmethod
    def _parse_duration_to_seconds(cls, duration_text):
        if not duration_text:
            return 0
        text = duration_text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\bmins?\b", "minutes", text)
        text = re.sub(r"\bhrs?\b",  "hours",   text)
        text = re.sub(r"\s+", " ", text).strip()
        units = {"second": 1, "seconds": 1,
                 "minute": 60, "minutes": 60,
                 "hour": 3600, "hours": 3600}
        total, qty = 0, []
        for token in text.split():
            if token in units:
                total += int((cls._parse_quantity(qty) or 1) * units[token])
                qty = []
            else:
                qty.append(token)
        return total

    @staticmethod
    def _parse_quantity(tokens):
        if not tokens:
            return None
        tokens = [t for t in tokens if t != "of"]
        if not tokens:
            return None
        if "half" in tokens:
            prefix = [t for t in tokens if t not in {"half", "and"}]
            return (GameView._parse_simple_number(prefix) or 0) + 0.5
        if "quarter" in tokens or "quarters" in tokens:
            q = [t for t in tokens if t not in {"a", "an", "and"}]
            if q in (["quarter"], ["quarters"]):
                return 0.25
            if q[-1] in {"quarter", "quarters"}:
                return (GameView._parse_simple_number(q[:-1]) or 1) * 0.25
        return GameView._parse_simple_number([t for t in tokens if t != "and"])

    @staticmethod
    def _parse_simple_number(tokens):
        if not tokens:
            return None
        nw = {
            "a": 1, "an": 1, "zero": 0, "one": 1, "two": 2, "three": 3,
            "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
            "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
            "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90,
        }
        total, found = 0, False
        for t in tokens:
            if t.isdigit():
                total += int(t); found = True
            elif t in nw:
                total += nw[t]; found = True
        return total if found else None