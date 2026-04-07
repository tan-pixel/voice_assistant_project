import random

class AnswerGenerationModule:
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
            return f"The current temperature in {city} is {temp} degrees Celsius."
        elif source == "timer":
            return f"I have set a {data.get('name')} for {data.get('duration')}."
            
        elif source == "system":
            return data.get("message", "I didn't quite catch that.")
            
        return "I have no response for that."

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