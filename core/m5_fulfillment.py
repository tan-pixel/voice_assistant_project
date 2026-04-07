import requests
from domain.dnd_client import DnDClient
from control_system.game_engine import DungeonGameEngine

class FulfillmentModule:
    def __init__(self):
        self.dnd_client = DnDClient()
        self.game_engine = DungeonGameEngine()
        
        # Define which intents belong to which system
        self.dnd_intents = [
            "lookup_monster", "lookup_spell", "lookup_weapon", 
            "lookup_armor", "equipment_category"
        ]
        
        self.game_intents = [
            "move", "attack", "use_item", "equip_item", 
            "pick_up", "inspect_room", "show_inventory", 
            "show_status", "open_object", "restart_game"
        ]

    def process(self, intent_data):
        """
        Input: Dictionary containing 'intent' and 'slots'
        Output: JSON response from API or State Change dictionary
        """
        if not intent_data or "intent" not in intent_data:
            return {"error": "Invalid intent data provided to Fulfillment module."}

        intent = intent_data["intent"]
        slots = intent_data.get("slots", {})

        # Route to Weather API
        if intent == "Weather":
            print(f"Routing to Weather API for intent: {intent}")
            return self._fulfill_weather(intent, slots)
            
        # Route to Timer
        elif intent == "Timer":
            print(f"Routing to Timer for intent: {intent}")
            duration = slots.get("duration", "1 minute")
            name = slots.get("name", "Timer")
            return {"source": "timer", "data": {"duration": duration, "name": name}, "intent": intent}

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
        city = slots.get("city", "Ottawa") # Default to Ottawa if no city is caught
        
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