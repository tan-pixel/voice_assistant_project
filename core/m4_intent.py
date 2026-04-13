import os
import json
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class JointIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased", local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        
        # MLP Head for Intent
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
        # Standard Linear Head for Slots
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]
        
        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits

# The main Inference Module
class IntentDetectionModule:
    DND_INTENTS = {
        "lookup_monster", "lookup_spell", "equipment_category", "lookup_weapon",
        "lookup_armor", "lookup_class", "lookup_race", "lookup_condition",
    }
    GAME_INTENTS = {
        "move", "attack", "inspect_room", "pick_up", "use_item", "open_object", "restart_game",
    }
    WEATHER_KEYWORDS = {
        "weather", "temperature", "forecast", "rain", "raining", "snow", "snowing",
        "sunny", "cloudy", "fog", "foggy", "storm", "stormy", "clear", "overcast",
        "condition", "conditions", "wind", "windy", "gust", "gusty", "breeze", "breezy",
        "umbrella",
    }
    WEATHER_WIND_KEYWORDS = {"wind", "windy", "gust", "gusty", "breeze", "breezy"}
    WEATHER_RAIN_KEYWORDS = {"rain", "raining", "umbrella", "drizzle", "showers"}
    WEATHER_SNOW_KEYWORDS = {"snow", "snowing", "snowy", "blizzard"}
    WEATHER_CONDITION_KEYWORDS = {
        "condition", "conditions", "forecast",
        "sunny", "cloudy", "fog", "foggy", "storm", "stormy", "clear", "overcast",
    }
    ALARM_TRIGGER_PHRASES = (
        "alarm",
        "wake me",
        "wake up",
        "awake by",
        "alert me",
        "get me up",
        "dont let me sleep",
        "don't let me sleep",
    )
    HELP_EXACT_PHRASES = {
        "what can you do",
        "help me",
        "show me commands",
        "what are my options",
        "how does this game work",
        "what can i ask you",
        "give me instructions",
        "how do i play",
        "what actions are available",
        "show me how to use this",
        "i need help",
        "can you guide me",
        "what features do you have",
        "how does this system work",
        "what commands can i use",
    }
    HELP_KEYWORDS = {"help", "commands", "command", "instructions", "guide", "options", "features"}
    HELP_PATTERNS = (
        r"\bwhat can (?:you|i) do\b",
        r"\bwhat can i ask you\b",
        r"\bwhat are my options\b",
        r"\bhow do i play\b",
        r"\bhow does this (?:game|system|work)\b",
        r"\bwhat actions are available\b",
        r"\bshow me how to use this\b",
        r"\bwhat features do you have\b",
        r"\bwhat commands can i use\b",
    )
    GAME_RESTART_PHRASES = {
        "restart the game", "restart game", "start over", "reset the dungeon",
        "reset the game", "start fresh", "begin from scratch", "new game",
    }
    GAME_INSPECT_PHRASES = {
        "look around", "inspect the room", "search the room", "what is in here",
        "what is around me", "what is here", "scan the area", "inspect my surroundings",
        "show nearby objects", "check the room", "examine the room", "search this area",
        "take a look around", "tell me what is nearby", "scan this room",
    }
    GAME_ATTACK_KEYWORDS = {"attack", "hit", "kill", "strike", "fight", "slash", "stab", "smash", "slay"}
    GAME_PICKUP_KEYWORDS = {"pick", "grab", "take", "collect", "loot", "select", "lift", "cup", "get"}
    GAME_USE_KEYWORDS = {"use", "drink", "consume", "apply", "equip", "wear", "activate"}
    GAME_OPEN_KEYWORDS = {"open", "unlock"}
    GAME_MOVE_KEYWORDS = {"move", "go", "walk", "head", "run", "travel", "step", "proceed", "advance", "continue", "turn"}
    GAME_DIRECTION_WORDS = {"north", "south", "east", "west", "up", "down", "left", "right"}
    GAME_DIRECTION_ALIASES = {
        "rate": "right",
        "write": "right",
        "wright": "right",
        "lift": "left",
    }
    GAME_ITEM_ALIASES = {
        "health potion": "Health Potion",
        "potion": "Health Potion",
        "rusty key": "Rusty Key",
        "key": "Rusty Key",
        "battle axe": "Battle Axe",
        "axe": "Battle Axe",
        "iron sword": "Iron Sword",
        "sword": "Iron Sword",
    }
    GAME_ITEM_KEYWORD_ALIASES = {
        "potion": "Health Potion",
        "health": "Health Potion",
        "key": "Rusty Key",
        "sword": "Iron Sword",
        "axe": "Battle Axe",
        "weapon": "Battle Axe",
    }
    GAME_OBJECT_ALIASES = {
        "treasure chest": "Treasure Chest",
        "locked chest": "Treasure Chest",
        "loot chest": "Treasure Chest",
        "prize chest": "Treasure Chest",
        "chest": "Treasure Chest",
        "treasure": "Treasure Chest",
        "box": "Treasure Chest",
    }
    GAME_MONSTER_ALIASES = {
        "goblin": "Goblin",
        "monster": "Goblin",
        "enemy": "Goblin",
        "creature": "Goblin",
        "beast": "Goblin",
    }
    DND_CLASS_NAMES = (
        "barbarian", "bard", "cleric", "druid", "fighter", "monk",
        "paladin", "ranger", "rogue", "sorcerer", "warlock", "wizard",
    )
    DND_RACE_NAMES = (
        "dragonborn", "half orc", "half elf", "halfling", "tiefling",
        "dwarf", "elf", "gnome", "human",
    )
    DND_CONDITION_NAMES = (
        "unconscious", "restrained", "petrified", "poisoned", "charmed",
        "blinded", "stunned", "invisible", "grappled", "prone",
    )
    DND_WEAPON_NAMES = (
        "long sword", "short sword", "cross bow", "battle axe",
        "heavy crossbow", "light crossbow", "greatsword", "battleaxe",
        "warhammer", "shortbow", "longsword", "handaxe", "dagger", "rapier",
    )
    DND_ARMOR_NAMES = (
        "studded leather", "chain mail", "half plate", "ring mail",
        "scale mail", "leather armor", "plate armor", "padded armor",
        "splint armor", "shield",
    )
    DND_CATEGORY_NAMES = (
        "martial weapons", "simple weapons", "ranged weapons", "melee weapons",
        "heavy armor", "light armor", "medium armor", "shields", "tools",
        "adventuring gear",
    )

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
        
        # Define paths where offline-trained models will be saved
        self.model_path = "data/models/joint_bert.pt"
        self.intent_map_path = "data/models/intent2id.json"
        self.slot_map_path = "data/models/slot2id.json"
        
        self.model_loaded = self._load_model()

    def _load_model(self):
        """Attempts to load the trained PyTorch model and mappings."""
        if not (os.path.exists(self.model_path) and os.path.exists(self.intent_map_path) and os.path.exists(self.slot_map_path)):
            print("WARNING: Trained BERT model files not found. Using fallback heuristics.")
            return False
        
        with open(self.intent_map_path, 'r') as f:
            self.intent2id = json.load(f)
            
        with open(self.slot_map_path, 'r') as f:
            self.slot2id = json.load(f)
        
        # Dynamic dimension checking 
        state_dict = torch.load(self.model_path, map_location=self.device)
        model_intents = state_dict["intent_classifier.3.weight"].shape[0]
        model_slots = state_dict["slot_classifier.weight"].shape[0]

        self.id2intent = {
            idx: name for name, idx in self.intent2id.items() if idx < model_intents
        }
        self.id2slot = {
            idx: name for name, idx in self.slot2id.items() if idx < model_slots
        }

        if len(self.id2intent) != model_intents or len(self.id2slot) != model_slots:
            raise ValueError("Saved intent/slot mappings do not cover the loaded model dimensions.")

        self.model = JointIntentSlotModel(model_intents, model_slots)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Successfully loaded Joint Intent & Slot BERT model.")
        return True

    def process(self, transcribed_text):
        """
        Input: Sentence from ASR module
        Output: Recognized Intent and required Slots
        """
        text = transcribed_text.lower().strip()
        
        if self.model_loaded:
            bert_result = self._predict_with_bert(text)
            result = self._apply_help_intent_rules(text, bert_result)
            result = self._apply_alarm_intent_rules(text, result)
            result = self._apply_weather_intent_rules(text, result)
            result = self._apply_game_intent_rules(text, result)
            result = self._apply_dnd_intent_rules(text, result)
            result["raw_text"] = text
            return result
        else:
            raise FileNotFoundError("Trained BERT model not found. Please run the training script to generate the model files.")

    def _apply_weather_intent_rules(self, text, bert_result):
        if not self._looks_like_weather_request(text, bert_result):
            return bert_result

        override_intent = self._classify_weather_intent(text)
        if override_intent is None:
            return bert_result

        
        slots = dict(bert_result.get("slots", {}))
        if "city" not in slots and "location" in slots:
            slots["city"] = slots["location"]
        if "city" not in slots:
            city = self._extract_city_from_text(text)
            if city:
                slots["city"] = city

        return {"intent": override_intent, "slots": slots}

    def _apply_dnd_intent_rules(self, text, bert_result):
        slots = dict(bert_result.get("slots", {}))
        override_intent = self._classify_dnd_intent(text)

        if override_intent is not None:
            extracted = self._extract_dnd_slot(text, override_intent)
            if extracted:
                slots[self._slot_name_for_dnd_intent(override_intent)] = extracted
            return {"intent": override_intent, "slots": slots}

        if bert_result.get("intent") in self.DND_INTENTS:
            intent = bert_result["intent"]
            extracted = self._extract_dnd_slot(text, intent)
            if extracted:
                slots.setdefault(self._slot_name_for_dnd_intent(intent), extracted)
            return {"intent": intent, "slots": slots}

        return bert_result

    def _apply_alarm_intent_rules(self, text, bert_result):
        if not self._looks_like_alarm_request(text, bert_result):
            return bert_result

        slots = dict(bert_result.get("slots", {}))
        return {"intent": "Alarm", "slots": slots}

    def _apply_help_intent_rules(self, text, bert_result):
        if not self._looks_like_help_request(text, bert_result):
            return bert_result
        return {"intent": "Help", "slots": {}}

    def _apply_game_intent_rules(self, text, bert_result):
        slots = dict(bert_result.get("slots", {}))
        override_intent = self._classify_game_intent(text, bert_result)
        intent = override_intent or bert_result.get("intent")

        if intent not in self.GAME_INTENTS:
            return bert_result

        slots.update(self._extract_game_slots(text, intent, slots))
        return {"intent": intent, "slots": slots}

    def _looks_like_weather_request(self, text, bert_result):
        if bert_result.get("intent") == "Weather":
            return True
        words = set(re.findall(r"[a-z]+", text))
        if any(word in (self.WEATHER_KEYWORDS - {"condition", "conditions"}) for word in words):
            return True
        if {"condition", "conditions"} & words:
            return any(word in words for word in {"weather", "sky", "sunny", "cloudy", "clear", "foggy", "stormy", "overcast"})
        return False

    def _looks_like_alarm_request(self, text, bert_result):
        if bert_result.get("intent") == "Alarm":
            return True
        if any(phrase in text for phrase in self.ALARM_TRIGGER_PHRASES):
            return True

        # Support time-of-day requests such as "set for 7 am tomorrow"
        # without confusing plain duration timers like "set a timer for 5 minutes".
        has_time_of_day = bool(
            re.search(r"\b\d{1,2}(?::|\s)?\d{0,2}\s*(am|pm)\b", text)
            or re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b(?:\s+(fifteen|thirty))?\s*(am|pm|o'clock)\b", text)
            or re.search(r"\bquarter\s+past\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", text)
        )
        has_day_hint = bool(re.search(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next|this|every)\b", text))
        looks_like_timer = bool(re.search(r"\b(timer|countdown|seconds?|minutes?|hours?)\b", text))
        return has_time_of_day and has_day_hint and not looks_like_timer

    def _looks_like_help_request(self, text, bert_result):
        if bert_result.get("intent") == "Help":
            return True
        if text in self.HELP_EXACT_PHRASES:
            return True
        if any(re.search(pattern, text) for pattern in self.HELP_PATTERNS):
            return True
        if any(keyword in text for keyword in self.HELP_KEYWORDS):
            return "timer" not in text and "alarm" not in text
        return False

    def _classify_game_intent(self, text, bert_result):
        words = set(re.findall(r"[a-z]+", text))
        item_name = self._extract_game_item(text)

        if any(phrase in text for phrase in self.GAME_RESTART_PHRASES):
            return "restart_game"
        if any(phrase in text for phrase in self.GAME_INSPECT_PHRASES):
            return "inspect_room"
        if words & self.GAME_OPEN_KEYWORDS and self._extract_game_object(text):
            return "open_object"
        if words & self.GAME_ATTACK_KEYWORDS and self._extract_game_monster(text):
            return "attack"
        if item_name and re.search(r"\bpick\s+up\b", text):
            return "pick_up"
        if words & self.GAME_PICKUP_KEYWORDS:
            return "pick_up"
        if item_name and not self._extract_direction_from_text(text):
            return "pick_up"
        if words & self.GAME_USE_KEYWORDS:
            return "use_item"
        if self._extract_direction_from_text(text) and (words & self.GAME_MOVE_KEYWORDS or words & self.GAME_DIRECTION_WORDS):
            return "move"
        if words & self.GAME_MOVE_KEYWORDS:
            return "move"
        if bert_result.get("intent") in (self.DND_INTENTS | {"OOS"}):
            if self._extract_game_object(text):
                return "open_object"
            if self._extract_game_monster(text):
                return "attack"
            if self._extract_game_item(text):
                return "pick_up"
        if bert_result.get("intent") in self.GAME_INTENTS:
            return bert_result["intent"]
        return None

    def _extract_game_slots(self, text, intent, slots):
        extracted = {}

        if intent == "move":
            direction = slots.get("direction") or self._extract_direction_from_text(text)
            if direction:
                extracted["direction"] = direction
        elif intent in {"pick_up", "use_item"}:
            item_name = slots.get("item_name") or self._extract_game_item(text)
            if item_name:
                extracted["item_name"] = item_name
        elif intent == "attack":
            monster_name = slots.get("monster_name") or self._extract_game_monster(text)
            if monster_name:
                extracted["monster_name"] = monster_name
        elif intent == "open_object":
            object_name = slots.get("object_name") or self._extract_game_object(text)
            if object_name:
                extracted["object_name"] = object_name

        return extracted

    def _extract_game_item(self, text):
        exact = self._extract_alias_value(text, self.GAME_ITEM_ALIASES)
        if exact:
            return exact
        words = set(re.findall(r"[a-z]+", text))
        for keyword, canonical in self.GAME_ITEM_KEYWORD_ALIASES.items():
            if keyword in words:
                return canonical
        return None

    def _extract_game_object(self, text):
        return self._extract_alias_value(text, self.GAME_OBJECT_ALIASES)

    def _extract_game_monster(self, text):
        return self._extract_alias_value(text, self.GAME_MONSTER_ALIASES)

    @classmethod
    def _extract_alias_value(cls, text, alias_map):
        for alias in sorted(alias_map, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return alias_map[alias]
        return None

    @classmethod
    def _extract_direction_from_text(cls, text):
        for direction in ("north", "south", "east", "west", "up", "down", "left", "right"):
            if re.search(rf"\b{direction}\b", text):
                return direction
        if re.search(r"\b(move|go|walk|head|run|travel|step|proceed|advance|continue|turn)\b", text):
            for alias, direction in cls.GAME_DIRECTION_ALIASES.items():
                if re.search(rf"\b{alias}\b", text):
                    return direction
        return None

    def _classify_dnd_intent(self, text):
        if self._find_known_name(text, self.DND_CLASS_NAMES) and "class" in text:
            return "lookup_class"
        if self._find_known_name(text, self.DND_RACE_NAMES) and "race" in text:
            return "lookup_race"
        if self._find_known_name(text, self.DND_CONDITION_NAMES) and ("condition" in text or "what happens" in text or "what does" in text):
            return "lookup_condition"
        if self._find_known_name(text, self.DND_CATEGORY_NAMES) or "equipment category" in text:
            return "equipment_category"
        if self._find_known_name(text, self.DND_WEAPON_NAMES) and (
            "weapon" in text
            or "damage" in text
            or "properties" in text
            or "stats" in text
            or re.search(r"\bwhat is\b", text)
            or re.search(r"\btell me about\b", text)
            or re.search(r"\bdescribe\b", text)
        ):
            return "lookup_weapon"
        if self._find_known_name(text, self.DND_ARMOR_NAMES) and ("armor" in text or "armour" in text or "shield" in text or "armor class" in text):
            return "lookup_armor"
        return None

    def _extract_dnd_slot(self, text, intent):
        if intent == "lookup_class":
            return self._find_known_name(text, self.DND_CLASS_NAMES)
        if intent == "lookup_race":
            return self._find_known_name(text, self.DND_RACE_NAMES)
        if intent == "lookup_condition":
            return self._find_known_name(text, self.DND_CONDITION_NAMES)
        if intent == "equipment_category":
            return self._find_known_name(text, self.DND_CATEGORY_NAMES)
        if intent == "lookup_weapon":
            return self._find_known_name(text, self.DND_WEAPON_NAMES) or self._extract_tail_entity(text, {"weapon"})
        if intent == "lookup_armor":
            return self._find_known_name(text, self.DND_ARMOR_NAMES) or self._extract_tail_entity(text, {"armor", "armour", "shield"})
        return None

    @staticmethod
    def _slot_name_for_dnd_intent(intent):
        return {
            "lookup_monster": "monster_name",
            "lookup_spell": "spell_name",
            "equipment_category": "category",
            "lookup_weapon": "weapon_name",
            "lookup_armor": "armor_name",
            "lookup_class": "class_name",
            "lookup_race": "race_name",
            "lookup_condition": "condition_name",
        }.get(intent, "")

    @staticmethod
    def _find_known_name(text, options):
        text = text.lower()
        matches = [option for option in options if option in text]
        if not matches:
            return None
        return max(matches, key=len)

    @staticmethod
    def _extract_tail_entity(text, keywords):
        patterns = [
            r"\babout\s+(?:the\s+|a\s+)?([a-z][a-z\s'\-]+?)(?:\s+(?:weapon|armor|armour))?$",
            r"\bdetails\s+for\s+(?:the\s+|a\s+)?([a-z][a-z\s'\-]+?)(?:\s+(?:weapon|armor|armour))?$",
            r"\bstats\s+for\s+(?:the\s+|a\s+)?([a-z][a-z\s'\-]+?)(?:\s+(?:weapon|armor|armour))?$",
            r"\blook\s+up\s+(?:the\s+|a\s+)?([a-z][a-z\s'\-]+?)(?:\s+(?:weapon|armor|armour))?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip(" ,.?!")
                if value and value not in keywords:
                    return value
        return None

    def _classify_weather_intent(self, text):
        words = set(re.findall(r"[a-z]+", text))
        if any(word in self.WEATHER_WIND_KEYWORDS for word in words):
            return "WeatherWind"
        if any(word in self.WEATHER_RAIN_KEYWORDS for word in words):
            return "WeatherRain"
        if any(word in self.WEATHER_SNOW_KEYWORDS for word in words):
            return "WeatherSnow"
        if any(word in self.WEATHER_CONDITION_KEYWORDS for word in words):
            return "WeatherCondition"
        if "weather" in words or "temperature" in words:
            return "Weather"
        return None

    @staticmethod
    def _extract_city_from_text(text):
        patterns = [
            r"\bin\s+([a-zA-Z][a-zA-Z\s'\-]+?)(?:\s+today|\s+right now|\s+now)?$",
            r"\bfor\s+([a-zA-Z][a-zA-Z\s'\-]+?)(?:\s+today|\s+right now|\s+now)?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                city = match.group(1).strip(" ,.?!")
                if city:
                    return city
        return None
        
    def _predict_with_bert(self, text):
        """Runs the actual DistilBERT inference on the input text."""
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=32).to(self.device)
        
        with torch.no_grad():
            intent_logits, slot_logits = self.model(encoding["input_ids"], encoding["attention_mask"])
            
            # Predict Intent
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            intent_name = self.id2intent[intent_idx]
            
            # Predict Slots
            slot_preds_idx = torch.argmax(slot_logits, dim=2)[0]
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            
            slots = {}
            current_slot_name = None
            current_slot_value = []
            
            # Extract BIO tags into a dictionary
            for idx, p_idx in enumerate(slot_preds_idx):
                # Skip special tokens ([CLS], [SEP], padding)
                if tokens[idx] in ["[CLS]", "[SEP]", "[PAD]"]: 
                    continue
                    
                slot_label = self.id2slot[p_idx.item()]
                token = tokens[idx]
                
                # Reconstruct words split by the WordPiece tokenizer (e.g., "##ing")
                if token.startswith("##"):
                    token = token[2:]
                    
                if slot_label.startswith("B-"):
                    if current_slot_name:
                        slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()
                    current_slot_name = slot_label[2:].lower()
                    current_slot_value = [token]
                elif slot_label.startswith("I-") and current_slot_name == slot_label[2:].lower():
                    current_slot_value.append(f" {token}")
                else:
                    if current_slot_name:
                        slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()
                        current_slot_name = None
                        current_slot_value = []
            
            # Catch the last slot if the sentence ended on an entity
            if current_slot_name:
                slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()

            return {"intent": intent_name, "slots": slots}
