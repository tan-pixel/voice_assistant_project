import random
import ollama

class AnswerGenerationModule:
    
    def __init__(self, use_llm=True, ollama_model="qwen2.5:1.5b-instruct"):
        self.use_llm = use_llm
        self.ollama_model = ollama_model
        
        if self.use_llm:
            print(f"Connecting to Local LLM ({self.ollama_model}) via Ollama package...")
            try:
                # Ping the server by trying to list the downloaded models
                ollama.list()
            except Exception as e:
                print(f"Notice: Ollama background server is not running ({e}). Falling back to Template Generation.")
                self.use_llm = False
    
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

        # Ollama LLM route
        if self.use_llm:
            return self._generate_with_ollama(source, intent, data)
        
        # Template route
        else:
            return self._generate_with_templates(source, intent, data)

    def _generate_with_ollama(self, source, intent, data):
        """Creates a prompt from the JSON and asks Ollama to generate the response."""
        
        # Few-Shot Prompting
        system_prompt = """You are Atlas, a conversational voice assistant. 
        Your ONLY job is to convert the provided Intent and JSON Data into a single, natural, spoken English sentence.

        CRITICAL RULES:
        1. NEVER explain the JSON structure or say "This is a dictionary/JSON".
        2. NEVER use markdown, asterisks, or bullet points.
        3. Output EXACTLY ONE conversational sentence. Stop talking immediately after.

        EXAMPLE 1:
        Intent: Weather
        Data: {'city': 'Paris', 'current': {'temperature_2m': 15, 'weather_code': 3}}
        Output: The current temperature in Paris is 15 degrees.

        EXAMPLE 2:
        Intent: lookup_monster
        Data: {'name': 'Ancient Red Dragon', 'armor_class': 22, 'hit_points': 546}
        Output: An Ancient Red Dragon is a terrifying foe with an armor class of 22 and 546 hit points.
        """

        # User Prompt
        user_prompt = f"Intent: {intent}\nData: {data}\nOutput:"
        
        try:
            # We pass the system instructions separately from the user prompt for better obedience
            response = ollama.generate(
                model=self.ollama_model, 
                system=system_prompt,
                prompt=user_prompt
            )
            
            # Extract the text 
            result_text = response.get("response", "")
            
            # Clean up the response to ensure TTS reads it nicely
            clean_text = result_text.strip().replace("*", "").replace("\n", " ")
            
            # # If the model still babbles, cut it off at the first period.
            # if "." in clean_text:
            #     clean_text = clean_text.split(".")[0] + "."
                
            return clean_text
            
        except Exception as e:
            print(f"Ollama Generation Failed: {e}. Falling back to templates.")
            return self._generate_with_templates(source, intent, data)
        
    def _generate_with_templates(self, source, intent, data):
        """Hardcoded fallback templates."""
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
            if intent == "Greetings":
                return "Hello! I am Atlas. What can I do for you today?"
            elif intent == "Goodbye":
                return "Goodbye! Have a great adventure."
            elif intent == "OOS":
                return "I'm sorry, I don't know how to do that yet."
            else:
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