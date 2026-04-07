class DungeonGameEngine:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        """Initializes the starting state of the simulated game."""
        self.state = {
            "position": [6, 3], # [y, x]
            "hp": 78,
            "max_hp": 100,
            "weapon": "Sword",
            "inventory": ["Health Potion", "Rusty Key"],
            "map_items": {
                "6,2": "Health Potion", # Dropped item
                "3,3": "Treasure Chest",
                "5,5": "Goblin"
            }
        }

    def fulfill_game_intent(self, intent, slots):
        """Updates game state based on intents and returns a state diff/message."""
        
        if intent == "move":
            direction = slots.get("direction", "").lower()
            return self._move_player(direction)
            
        elif intent == "pick_up":
            item = slots.get("item_name", "").title()
            return self._pick_up_item(item)
            
        elif intent == "use_item":
            item = slots.get("item_name", "").title()
            return self._use_item(item)
            
        elif intent == "restart_game":
            self.reset_game()
            return {"status": "success", "message": "The dungeon shifts... Game restarted.", "state": self.state}
            
        else:
            return {"status": "error", "message": f"Game intent '{intent}' not recognized."}

    def _move_player(self, direction):
        """Updates position."""
        y, x = self.state["position"]
        if direction == "north" or direction == "up": y -= 1
        elif direction == "south" or direction == "down": y += 1
        elif direction == "east" or direction == "right": x += 1
        elif direction == "west" or direction == "left": x -= 1
        else:
            return {"status": "error", "message": "I don't know which way that is."}

        self.state["position"] = [y, x]
        return {"status": "success", "message": f"You moved {direction.title()}.", "state": self.state}

    def _pick_up_item(self, item):
        """Adds item to inventory and removes it from the map."""
        pos_key = f"{self.state['position'][0]},{self.state['position'][1]}"
        
        if self.state["map_items"].get(pos_key) == item:
            self.state["inventory"].append(item)
            del self.state["map_items"][pos_key]
            return {"status": "success", "message": f"You picked up the {item}.", "state": self.state}
        else:
            return {"status": "error", "message": f"There is no {item} here."}

    def _use_item(self, item):
        """Consumes an item and applies effects (e.g., healing)."""
        if item in self.state["inventory"]:
            if item == "Health Potion":
                self.state["inventory"].remove(item)
                heal_amount = 20
                self.state["hp"] = min(self.state["hp"] + heal_amount, self.state["max_hp"])
                return {"status": "success", "message": f"You drank a potion and recovered HP.", "state": self.state}
            else:
                return {"status": "error", "message": f"You cannot use the {item} right now."}
        else:
            return {"status": "error", "message": f"You don't have a {item} in your inventory."}