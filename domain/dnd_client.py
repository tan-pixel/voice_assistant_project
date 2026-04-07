import requests

class DnDClient:
    def __init__(self):
        self.base_url = "https://www.dnd5eapi.co/api"

    def fulfill_dnd_intent(self, intent, slots):
        """Routes the D&D intent to the correct API endpoint."""
        
        if intent == "lookup_monster":
            monster_name = slots.get("monster_name", "").lower().replace(" ", "-")
            return self._get_request(f"/monsters/{monster_name}")
            
        elif intent == "lookup_spell":
            spell_name = slots.get("spell_name", "").lower().replace(" ", "-")
            return self._get_request(f"/spells/{spell_name}")
            
        elif intent == "equipment_category":
            category = slots.get("category", "").lower().replace(" ", "-")
            return self._get_request(f"/equipment-categories/{category}")
            
        else:
            return {"error": f"D&D Intent '{intent}' not implemented yet."}

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