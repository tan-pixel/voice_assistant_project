"""
voice_assistant_ui.py  –  Atlas Voice Assistant (Group 15)

Dark mode UI with green accents. Map stays light.
Requires:  pip install ttkbootstrap
"""

import threading
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from ui.pipeline_view import PipelineView
from ui.game_view import GameView


# ── dark palette with green accents ──────────────────────────────────────────
BG_ROOT   = "#141a16"   # near-black with a green undertone
BG_CARD   = "#1e2822"   # dark green-tinted card surface
BG_PANEL  = "#263020"   # slightly lighter panel interior
BORDER    = "#3a5040"   # muted green border
ACCENT    = "#3dba72"   # bright sage green for headings / icons
FG_HEAD   = "#e8f5ed"   # off-white heading text
FG_BODY   = "#b8d4c0"   # soft green-grey body text
FG_DIM    = "#5a7a62"   # muted for placeholders / step labels

FONT_H2   = ("Segoe UI", 10, "bold")
FONT_BODY = ("Segoe UI", 9)


class VoiceAssistantUI:
    def __init__(self, root, orchestrator):
        self.root = root
        self.orchestrator = orchestrator

        self.root.title("🎙️  Atlas Voice Assistant  ·  Group 15")
        self.root.geometry("1140x680")
        self.root.minsize(900, 540)
        self.root.configure(bg=BG_ROOT)

        # "darkly" is ttkbootstrap's clean dark base; we layer our green palette on top
        self._style = ttk.Style(theme="darkly")
        self._patch_styles()

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        left = self._card(self.root, col=0, padx=(12, 6))

        self._heading(left, "⚡  Pipeline")
        self._rule(left)
        self.pipeline_view = PipelineView(left, self.orchestrator, self.update_game_log)

        self._heading(left, "🔧  Bypasses", pady=(18, 4))
        self._rule(left)
        self._build_bypass_panel(left)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        right = self._card(self.root, col=1, padx=(6, 12))

        self._heading(right, "🎮  Dungeon Crawler")
        self._rule(right)

        game_container = tk.Frame(right, bg=BG_CARD)
        game_container.pack(fill="both", expand=True, padx=4, pady=(0, 6))

        self.game_view = GameView(game_container)
        self.game_view.update_game_visuals(self.orchestrator.m5.game_engine.state)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _patch_styles(self):
        s = self._style
        s.configure("TFrame",    background=BG_CARD)
        s.configure("TLabel",    background=BG_CARD, foreground=FG_BODY, font=FONT_BODY)
        s.configure("TEntry",    padding=(6, 4),     font=FONT_BODY,
                                 fieldbackground=BG_PANEL, foreground=FG_BODY)
        s.configure("TButton",   font=FONT_BODY,     padding=(8, 4))
        s.configure("TCombobox", font=FONT_BODY,     padding=(4, 4),
                                 fieldbackground=BG_PANEL, foreground=FG_BODY)

    def _card(self, parent, col, padx=(12, 12)):
        f = tk.Frame(parent, bg=BG_CARD)
        f.grid(row=0, column=col, sticky="nsew", padx=padx, pady=12)
        f.columnconfigure(0, weight=1)
        return f

    def _heading(self, parent, text, pady=(10, 2)):
        tk.Label(parent, text=text, font=FONT_H2,
                 fg=ACCENT, bg=BG_CARD, anchor="w"
                 ).pack(fill="x", padx=14, pady=pady)

    def _rule(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 8))

    def _step_label(self, parent, text):
        tk.Label(parent, text=text, font=FONT_BODY,
                 fg=FG_DIM, bg=BG_CARD, anchor="w"
                 ).pack(fill="x", padx=14, pady=(9, 2))

    def _row(self, parent):
        f = tk.Frame(parent, bg=BG_CARD)
        f.pack(fill="x", padx=12, pady=3)
        return f

    def _entry_btn(self, parent, placeholder, btn_text, btn_style, cmd):
        row = self._row(parent)
        e = ttk.Entry(row, font=FONT_BODY, width=16)
        e.insert(0, placeholder)
        e.configure(foreground=FG_DIM)

        def fi(_):
            if e.get() == placeholder:
                e.delete(0, "end"); e.configure(foreground=FG_HEAD)
        def fo(_):
            if not e.get():
                e.insert(0, placeholder); e.configure(foreground=FG_DIM)

        e.bind("<FocusIn>", fi); e.bind("<FocusOut>", fo)
        e.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row, text=btn_text, bootstyle=btn_style,
                   width=9, command=cmd).pack(side="right")
        return e

    # ── bypass panel ──────────────────────────────────────────────────────────

    def _build_bypass_panel(self, parent):
        self._step_label(parent, "① Verification")
        self.code_input = self._entry_btn(
            parent, "passcode", "Unlock 🔑", "warning-outline",
            lambda: self.trigger_bypass(1, self._val(self.code_input, "passcode")))

        self._step_label(parent, "② Wake Word")
        self.wake_input = self._entry_btn(
            parent, "e.g. 'Hey Atlas'", "Wake 🔔", "info-outline",
            lambda: self.trigger_bypass(2, self._val(self.wake_input, "e.g. 'Hey Atlas'")))

        self._step_label(parent, "③ ASR  (type command)")
        self.asr_input = self._entry_btn(
            parent, "e.g. 'go north'", "Send 🎤", "success-outline",
            lambda: self.trigger_bypass(3, self._val(self.asr_input, "e.g. 'go north'")))

        self._step_label(parent, "④ Intent & Slots")
        self.intent_combo = ttk.Combobox(
            parent,
            values=["Weather", "Timer", "move", "lookup_monster", "Greetings"],
            state="readonly", font=FONT_BODY,
        )
        self.intent_combo.set("Weather")
        self.intent_combo.pack(fill="x", padx=12, pady=(2, 4))
        self.intent_combo.bind("<<ComboboxSelected>>", self._on_intent_changed)

        row4 = self._row(parent)
        self.slot_input = ttk.Entry(row4, font=FONT_BODY, width=16)
        self.slot_input.insert(0, "city: Paris")
        self.slot_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row4, text="Inject 💉", bootstyle="success-outline",
                   width=9, command=self._inject_intent_bypass).pack(side="right")

        self._step_label(parent, "⑤ ⑥ Pre-canned")
        row56 = self._row(parent)
        ttk.Button(row56, text="⑤  Canned API", bootstyle="secondary-outline",
                   command=lambda: self.trigger_bypass(5, None)
                   ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row56, text="⑥  Canned NL", bootstyle="secondary-outline",
                   command=lambda: self.trigger_bypass(6, None)
                   ).pack(side="right", fill="x", expand=True)

    # ── events ────────────────────────────────────────────────────────────────

    @staticmethod
    def _val(entry, placeholder):
        v = entry.get()
        return "" if v == placeholder else v

    def _on_intent_changed(self, _):
        self.slot_input.delete(0, "end")
        self.slot_input.insert(0, {
            "Weather": "city: Paris", "Timer": "duration: 5 minutes",
            "move": "direction: north", "lookup_monster": "monster_name: badger",
            "Greetings": "",
        }.get(self.intent_combo.get(), ""))

    def _inject_intent_bypass(self):
        intent, slot_text = self.intent_combo.get(), self.slot_input.get()
        slots = {}
        if ":" in slot_text:
            k, v = slot_text.split(":", 1)
            slots[k.strip()] = v.strip()
        self.trigger_bypass(4, {"intent": intent, "slots": slots})

    def trigger_bypass(self, step, data):
        threading.Thread(target=self.orchestrator.manual_override,
                         args=(step, data), daemon=True).start()

    def update_game_log(self, text):
        self.game_view.update_log(text)