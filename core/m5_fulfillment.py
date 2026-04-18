import re
from datetime import datetime, timedelta

import requests
from domain.dnd_client import DnDClient
from control_system.game_engine import DungeonGameEngine

class FulfillmentModule:
    WEEKDAYS = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    def __init__(self):
        self.dnd_client = DnDClient()
        self.game_engine = DungeonGameEngine()
        self.weather_intents = [
            "Weather",
            "WeatherWind",
            "WeatherCondition",
            "WeatherRain",
            "WeatherSnow",
        ]
        
        # Define which intents belong to which system
        self.dnd_intents = [
            "lookup_monster", "lookup_spell", "lookup_weapon",
            "lookup_armor", "equipment_category", "lookup_class",
            "lookup_race", "lookup_condition"
        ]
        
        self.game_intents = [
            "move",
            "attack",
            "inspect_room",
            "use_item",
            "pick_up",
            "open_object",
            "equip_item",
            "show_inventory",
            "show_status",
            "restart_game",
        ]

    def process(self, intent_data):
        """
        Input: Dictionary containing 'intent' and 'slots'
        Output: JSON response from API or State Change dictionary
        """
        if not intent_data or "intent" not in intent_data:
            return {"error": "Invalid intent data provided to Fulfillment module."}

        intent = intent_data["intent"]
        slots = self._normalize_slots(intent_data.get("slots", {}))
        raw_text = intent_data.get("raw_text", "")

        # Route to Weather API
        if intent in self.weather_intents:
            print(f"Routing to Weather API for intent: {intent}")
            return self._fulfill_weather(intent, slots)
            
        # Route to Timer
        elif intent == "Timer":
            print(f"Routing to Timer for intent: {intent}")
            duration = slots.get("duration", "1 minute")
            name = slots.get("name", "Timer")
            return {"source": "timer", "data": {"duration": duration, "name": name}, "intent": intent}

        # Route to Alarm
        elif intent == "Alarm":
            print(f"Routing to Alarm for intent: {intent}")
            return self._fulfill_alarm(slots, raw_text)

        elif intent == "Help":
            print(f"Routing to Help for intent: {intent}")
            return {"source": "system", "data": {"topic": "help"}, "intent": intent}

        # Route to D&D API
        elif intent in self.dnd_intents:
            print(f"Routing to D&D API for intent: {intent}")
            api_response = self.dnd_client.fulfill_dnd_intent(intent, slots)
            return {"source": "dnd_api", "data": api_response, "intent": intent}

        # Route to Game Engine
        elif intent in self.game_intents:
            print(f"Routing to Game Engine for intent: {intent}")
            game_response = self.game_engine.fulfill_game_intent(intent, slots)
            return {"source": "game_engine", "data": game_response, "intent": intent}

        # Handle Greetings/Goodbye/OOS
        else:
            return {"source": "system", "data": {"message": f"Intent '{intent}' is out of scope or a basic system command."}, "intent": intent}

    def _fulfill_weather(self, intent, slots):
        """Handles Open-Meteo API requests"""
        city = slots.get("city") or slots.get("location") or "Ottawa"  # Default to Ottawa if no location is caught
        
        # Add a User-Agent header to prevent SSL/EOF connection drops
        headers = {"User-Agent": "AtlasVoiceAssistant/1.0"}
        
        # Geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1}
        
        try:
            # Pass the headers
            geo_response = requests.get(geo_url, params=geo_params, headers=headers, timeout=5)
            geo_data = geo_response.json()
            
            if "results" not in geo_data:
                return {"source": "weather_api", "data": {"error": f"Could not find coordinates for {city}."}, "intent": intent}
                
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            
            # Weather Forecast
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "timezone": "auto"
            }
            
            # Pass the headers
            weather_response = requests.get(weather_url, params=weather_params, headers=headers, timeout=5)
            weather_data = weather_response.json()
            
            # Attach the city name to the data so M6 (Generation) can use it in the sentence
            weather_data["city"] = city.title()
            
            return {"source": "weather_api", "data": weather_data, "intent": intent}
            
        except Exception as e:
            return {"source": "weather_api", "data": {"error": str(e)}, "intent": intent}

    def _fulfill_alarm(self, slots, raw_text):
        name = slots.get("name") or self._extract_alarm_name(raw_text) or "Alarm"
        target_dt = self._parse_alarm_datetime(slots, raw_text)

        if target_dt is None:
            return {
                "source": "alarm",
                "data": {
                    "status": "error",
                    "message": "I couldn't understand when to set that alarm.",
                    "name": name,
                },
                "intent": "Alarm",
            }

        return {
            "source": "alarm",
            "data": {
                "name": name,
                "target_iso": target_dt.isoformat(),
                "target_display": self._format_alarm_target(target_dt),
            },
            "intent": "Alarm",
        }

    def _parse_alarm_datetime(self, slots, raw_text):
        raw_text = (raw_text or "").lower()
        now = datetime.now()
        time_components = self._parse_alarm_time(slots.get("time"), raw_text)
        if time_components is None:
            return None

        hour, minute = time_components
        explicit_day, target_date = self._parse_alarm_day(slots.get("day"), raw_text, now)
        target_dt = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if target_dt <= now:
            if explicit_day:
                target_dt += timedelta(days=1 if target_date.date() == now.date() else 7)
            else:
                target_dt += timedelta(days=1)

        return target_dt

    def _parse_alarm_day(self, day_text, raw_text, now):
        day_text = (day_text or "").lower()
        text = f"{raw_text} {day_text}".strip()

        if "tomorrow" in text:
            return True, now + timedelta(days=1)
        if "tonight" in text or "today" in text:
            return True, now

        for keyword in ("next", "this", "every"):
            match = re.search(rf"\b{keyword}\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
            if match:
                weekday = self.WEEKDAYS[match.group(1)]
                delta = (weekday - now.weekday()) % 7
                if keyword == "next":
                    delta = 7 if delta == 0 else delta + 7
                elif delta == 0 and keyword == "every":
                    delta = 7
                return True, now + timedelta(days=delta)

        match = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
        if match:
            weekday = self.WEEKDAYS[match.group(1)]
            delta = (weekday - now.weekday()) % 7
            return True, now + timedelta(days=delta)

        return False, now
    
    def _normalize_slots(self, slots):
        """Maps raw spoken words extracted by BERT to Canonical Database/Game IDs."""
        normalized = {}
        for key, value in slots.items():
            # Clean up the raw string
            val_lower = str(value).lower().strip()

            # Game Movement (Handle ASR typos)
            if key == "direction":
                aliases = {"rate": "right", "write": "right", "wright": "right", "lift": "left"}
                normalized[key] = aliases.get(val_lower, val_lower)
            
            # Game Items
            elif key == "item_name":
                aliases = {
                    "health potion": "Health Potion", "potion": "Health Potion", "health": "Health Potion",
                    "rusty key": "Rusty Key", "key": "Rusty Key",
                    "battle axe": "Battle Axe", "axe": "Battle Axe", "weapon": "Battle Axe",
                    "iron sword": "Iron Sword", "sword": "Iron Sword"
                }
                normalized[key] = aliases.get(val_lower, value.title())

            # Game Objects (Chests/Doors)
            elif key == "object_name":
                aliases = {
                    "treasure chest": "Treasure Chest", "locked chest": "Treasure Chest", 
                    "chest": "Treasure Chest", "box": "Treasure Chest", "treasure": "Treasure Chest"
                }
                normalized[key] = aliases.get(val_lower, value.title())
            
            # Game Enemies
            elif key == "monster_name":
                aliases = {"goblin": "Goblin", "monster": "Goblin", "enemy": "Goblin", "creature": "Goblin"}
                # If it's a D&D lookup, we want to format it for the web API instead of the game engine
                if "lookup" not in key: 
                    normalized[key] = aliases.get(val_lower, value.title())
                else:
                    normalized[key] = val_lower.replace(" ", "-").replace("'", "")
            
            # D&D 5e API Formatting
            # The D&D web API requires dashes instead of spaces (e.g. "ancient red dragon" -> "ancient-red-dragon")
            elif key in ["weapon_name", "spell_name", "class_name", "race_name", "armor_name", "condition_name", "category"]:
                normalized[key] = val_lower.replace(" ", "-").replace("'", "")
            
            # Default fallback
            else:
                normalized[key] = value

        return normalized

    def _parse_alarm_time(self, time_text, raw_text):
        search_text = f"{(time_text or '').lower()} {raw_text}".strip()

        match = re.search(r"\b(\d{1,2})\s*[-:]\s*(\d{2})\s*(am|pm)?\b", search_text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            meridiem = match.group(3)
            normalized = self._normalize_hour(hour, minute, meridiem)
            if normalized is not None:
                return normalized

        match = re.search(r"\b(?:at|for)\s+(\d{3,4})\s*(am|pm)?\b", search_text)
        if match:
            digits = match.group(1)
            hour = int(digits[:-2])
            minute = int(digits[-2:])
            meridiem = match.group(2)
            normalized = self._normalize_hour(hour, minute, meridiem)
            if normalized is not None:
                return normalized

        match = re.search(r"(?<![\d-])(\d{1,2})(?::|\s)?(\d{2})?\s*(am|pm)\b", search_text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            normalized = self._normalize_hour(hour, minute, match.group(3))
            if normalized is not None:
                return normalized

        match = re.search(r"\bat\s+(\d{1,2})(?::|\s)?(\d{2})?\b", search_text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            meridiem = self._infer_alarm_meridiem(hour, search_text)
            normalized = self._normalize_hour(hour, minute, meridiem)
            if normalized is not None:
                return normalized

        match = re.search(r"\bfor\s+(\d{1,2})(?::|\s)?(\d{2})?\b", search_text)
        if match and re.search(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next|this|every)\b", search_text):
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            meridiem = self._infer_alarm_meridiem(hour, search_text)
            normalized = self._normalize_hour(hour, minute, meridiem)
            if normalized is not None:
                return normalized

        match = re.search(r"\b(\d{1,2})\s+in\s+the\s+(morning|afternoon|evening|night)\b", search_text)
        if match:
            hour = int(match.group(1))
            minute = 0
            return self._normalize_hour(hour, minute, self._period_to_meridiem(match.group(2)))

        return self._parse_word_alarm_time(search_text)

    def _parse_word_alarm_time(self, text):
        number_words = {
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
        }
        minute_words = {"fifteen": 15, "thirty": 30}
        meridiem_hint = None

        if "morning" in text:
            meridiem_hint = "am"
        elif "afternoon" in text or "evening" in text or "night" in text:
            meridiem_hint = "pm"

        match = re.search(r"\bquarter\s+past\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b(?:\s+(am|pm))?", text)
        if match:
            hour = number_words[match.group(1)]
            meridiem = match.group(2) or meridiem_hint or "am"
            return self._normalize_hour(hour, 15, meridiem)

        match = re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
            r"(?:\s+(fifteen|thirty))?"
            r"(?:\s+(am|pm|o'clock))?",
            text,
        )
        if match:
            hour = number_words[match.group(1)]
            minute = minute_words.get(match.group(2), 0)
            meridiem = match.group(3)
            if meridiem == "o'clock":
                meridiem = meridiem_hint
            return self._normalize_hour(hour, minute, meridiem or meridiem_hint)

        return None

    @staticmethod
    def _normalize_hour(hour, minute, meridiem):
        if not (0 <= minute <= 59):
            return None

        if meridiem in {"am", "pm"}:
            # If ASR produced a 24-hour time plus am/pm ("20-58 pm"),
            # trust the 24-hour clock value and ignore the redundant meridiem.
            if 13 <= hour <= 23:
                return hour, minute
            if hour == 0:
                return 0, minute
            if 1 <= hour <= 12:
                normalized_hour = hour % 12
                if meridiem == "pm":
                    normalized_hour += 12
                return normalized_hour, minute
            return None

        if 0 <= hour <= 23:
            return hour, minute

        return None

    @staticmethod
    def _period_to_meridiem(period):
        return "am" if period == "morning" else "pm"

    @staticmethod
    def _infer_alarm_meridiem(hour, text):
        if re.search(r"\b(morning|am)\b", text):
            return "am"
        if re.search(r"\b(afternoon|evening|night|pm|tonight)\b", text):
            return "pm"
        # Demo-friendly default: earlier daytime hours usually mean morning.
        return "am" if 5 <= hour <= 11 else "pm"

    @staticmethod
    def _extract_alarm_name(raw_text):
        text = (raw_text or "").lower()
        stopwords = {"set", "create", "initialize", "put", "new", "a", "an"}

        match = re.search(r"\b(?:called|named|labeled)\s+([a-z][a-z\s']+?)(?:\s+(?:for|on|at|tomorrow|today|tonight|next|this|every)\b|$)", text)
        if match:
            return match.group(1).strip(" .?!")

        match = re.search(r"\bfor\s+(?:the\s+|my\s+)?([a-z][a-z\s']+)$", text)
        if match:
            return match.group(1).strip(" .?!")

        match = re.search(r"\b([a-z]+(?:\s+[a-z]+){0,2})\s+alarm\b", text)
        if match:
            candidate_tokens = [token for token in match.group(1).split() if token not in stopwords]
            if candidate_tokens:
                return " ".join(candidate_tokens).strip(" .?!")

        return None

    @staticmethod
    def _format_alarm_target(target_dt):
        return target_dt.strftime("%a %I:%M %p")
