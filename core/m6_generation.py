import random

class AnswerGenerationModule:
    HELP_RESPONSES = [
        "I can help with weather, timers, alarms, D and D lookups, and dungeon commands like move, pick up, use item, and restart game.",
        "You can ask for weather, set a timer or alarm, ask about D and D monsters, spells, or equipment, or play the dungeon with commands like move north and pick up the sword.",
        "Try things like what is the weather in Ottawa, set a tea timer for 5 minutes, set a work alarm for 7 AM tomorrow, tell me about a goblin, or move north.",
    ]
    WEATHER_CODE_DESCRIPTIONS = {
        0: "clear skies",
        1: "mostly clear skies",
        2: "partly cloudy skies",
        3: "overcast skies",
        45: "foggy conditions",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "light snowfall",
        73: "moderate snowfall",
        75: "heavy snowfall",
        77: "snow grains",
        80: "light rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "light snow showers",
        86: "heavy snow showers",
        95: "a thunderstorm",
        96: "a thunderstorm with light hail",
        99: "a thunderstorm with heavy hail",
    }
    RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
    SNOW_CODES = {71, 73, 75, 77, 85, 86}

    def process(self, fulfillment_data):
        """
        Input: Dictionary containing 'source', 'data', and 'intent' from M5.
        Output: A natural language string.
        """
        if not fulfillment_data or "error" in fulfillment_data:
            return "I'm sorry, I couldn't process that request."

        source = fulfillment_data.get("source")
        data = fulfillment_data.get("data", {})
        intent = fulfillment_data.get("intent")

        if source == "alarm":
            if data.get("status") == "error":
                return data.get("message", "I couldn't set that alarm.")
            return f"I have set your {data.get('name', 'alarm')} for {data.get('target_display', 'the requested time')}."

        # Log the real error to the terminal, but speak a polite sentence
        if "error" in data:
            print(f"Backend API Error: {data['error']}")
            return "I ran into a network issue and couldn't fetch that information right now."

        # Route to the correct generation logic
        if source == "game_engine":
            return self._generate_game_response(data)
        elif source == "dnd_api":
            return self._generate_dnd_response(intent, data)
        elif source == "weather_api":
            city = data.get("city", "that location")
            current = data.get("current", {})
            temp = current.get("temperature_2m", "unknown")
            wind = current.get("wind_speed_10m", "unknown")
            weather_code = current.get("weather_code")
            weather_description = self._describe_weather_code(weather_code)

            if intent == "WeatherWind":
                return f"The current wind speed in {city} is {wind} kilometers per hour."
            if intent == "WeatherCondition":
                return f"The current weather in {city} is {weather_description}."
            if intent == "WeatherRain":
                if self._is_rain_code(weather_code):
                    return f"Yes, it is currently rainy in {city}: {weather_description}."
                return f"No, it is not currently raining in {city}. The weather is {weather_description}."
            if intent == "WeatherSnow":
                if self._is_snow_code(weather_code):
                    return f"Yes, it is currently snowy in {city}: {weather_description}."
                return f"No, it is not currently snowing in {city}. The weather is {weather_description}."
            return f"The current temperature in {city} is {temp} degrees Celsius, with {weather_description}."
        elif source == "timer":
            return f"I have set a {data.get('name')} for {data.get('duration')}."
            
        elif source == "system":
            if intent == "Greetings":
                return "Hello! I am Atlas. What can I do for you today?"
            elif intent == "Goodbye":
                return "Goodbye! Have a great adventure."
            elif intent == "OOS":
                return "I'm sorry, I don't know how to do that yet."
            elif intent == "Help":
                return random.choice(self.HELP_RESPONSES)
            else:
                return data.get("message", "I didn't quite catch that.")
            
        return "I have no response for that."

    def _describe_weather_code(self, code):
        try:
            return self.WEATHER_CODE_DESCRIPTIONS.get(int(code), "unavailable conditions")
        except (TypeError, ValueError):
            return "unavailable conditions"

    def _is_rain_code(self, code):
        try:
            return int(code) in self.RAIN_CODES
        except (TypeError, ValueError):
            return False

    def _is_snow_code(self, code):
        try:
            return int(code) in self.SNOW_CODES
        except (TypeError, ValueError):
            return False

    def _generate_game_response(self, data):
        """Extracts the natural language message directly from the game engine state."""
        return data.get("message", "The dungeon state has been updated.")

    def _generate_dnd_response(self, intent, data):
        """Maps D&D API JSON fields to natural language templates."""
        
        try:
            if intent == "lookup_monster":
                name = data.get("name", "This monster")
                # Handle nested armor class list
                ac_list = data.get("armor_class", [])
                ac = ac_list[0].get("value") if ac_list else "unknown"
                hp = data.get("hit_points", "unknown")
                
                # Try to grab the first special ability
                abilities = data.get("special_abilities", [])
                if abilities:
                    ability_name = abilities[0].get("name")
                    ability_desc = abilities[0].get("desc")
                    return f"A {name} has an AC of {ac} and {hp} hit points. It can use {ability_name}: {ability_desc}"
                else:
                    return f"A {name} is a creature with an AC of {ac} and {hp} hit points."

            elif intent == "lookup_spell":
                name = data.get("name", "This spell")
                level = data.get("level", 0)
                school = data.get("school", {}).get("name", "magic")
                
                # Safely attempt to extract damage or effects
                damage_data = data.get("damage", {}).get("damage_at_slot_level", {})
                damage = damage_data.get(str(level)) if damage_data else "special effects"
                
                area = data.get("area_of_effect", {}).get("size", "a targeted")
                
                if damage != "special effects":
                    return f"{name} is a level {level} {school} spell that affects a {area}-foot area, dealing {damage} damage."
                else:
                    return f"{name} is a level {level} {school} spell. {data.get('desc', ['It has special effects.'])[0]}"

            elif intent == "equipment_category":
                category = data.get("name", "Equipment")
                items = data.get("equipment", [])
                
                if not items:
                    return f"I couldn't find any items in the {category} category."
                
                # Pick up to 4 random items to list so the VA doesn't read 50 weapons
                sample_items = [item.get("name") for item in items]
                if len(sample_items) > 4:
                    sample_items = random.sample(sample_items, 4)
                    
                item_list = ", ".join(sample_items[:-1]) + f", and {sample_items[-1]}"
                return f"{category} include a variety of options such as the {item_list}."

            else:
                return f"I found data for {intent}, but I'm not sure how to read it to you yet."
                
        except Exception as e:
            print(f"Generation Error: {e}")
            return "I found the information, but I had trouble formatting it into a sentence."
