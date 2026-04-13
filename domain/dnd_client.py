import requests

class DnDClient:
    def __init__(self):
        self.base_url = "https://www.dnd5eapi.co/api"

    def fulfill_dnd_intent(self, intent, slots):
        """Routes the D&D intent to the correct API endpoint."""
        
        if intent == "lookup_monster":
            return self._get_request(f"/monsters/{self._slugify(slots.get('monster_name', ''))}")
            
        elif intent == "lookup_spell":
            return self._get_request(f"/spells/{self._slugify(slots.get('spell_name', ''))}")
            
        elif intent == "equipment_category":
            return self._get_request(f"/equipment-categories/{self._slugify(slots.get('category', ''))}")

        elif intent == "lookup_weapon":
            return self._get_request(f"/equipment/{self._slugify(slots.get('weapon_name', ''))}")

        elif intent == "lookup_armor":
            return self._get_request(f"/equipment/{self._slugify(slots.get('armor_name', ''))}")

        elif intent == "lookup_class":
            return self._get_request(f"/classes/{self._slugify(slots.get('class_name', ''))}")

        elif intent == "lookup_race":
            return self._get_request(f"/races/{self._slugify(slots.get('race_name', ''))}")

        elif intent == "lookup_condition":
            return self._get_request(f"/conditions/{self._slugify(slots.get('condition_name', ''))}")
            
        else:
            return {"error": f"D&D Intent '{intent}' not implemented yet."}

    @staticmethod
    def _slugify(value):
        return value.lower().strip().replace("'", "").replace(" ", "-")

    def _get_request(self, endpoint):
        """Executes the GET request and returns JSON."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code} for {url}"}
        except Exception as e:
            return {"error": str(e)}
