import random
import re


class DungeonGameEngine:
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    CHEST_NAME = "Treasure Chest"
    OPEN_CHEST_NAME = "Opened Treasure"
    KEY_NAME = "Rusty Key"
    POTION_NAME = "Health Potion"
    MONSTER_NAME = "Goblin"
    WEAPON_DAMAGE = {
        "Rusty Dagger": (10, 14),
        "Iron Sword": (18, 24),
        "Battle Axe": (24, 30),
    }
    WEAPON_ITEMS = {"Iron Sword", "Battle Axe"}
    ITEM_ALIASES = {
        "health potion": "Health Potion",
        "potion": "Health Potion",
        "rusty key": "Rusty Key",
        "key": "Rusty Key",
        "battle axe": "Battle Axe",
        "axe": "Battle Axe",
        "iron sword": "Iron Sword",
        "sword": "Iron Sword",
    }
    ITEM_KEYWORD_ALIASES = {
        "potion": "Health Potion",
        "health": "Health Potion",
        "key": "Rusty Key",
        "sword": "Iron Sword",
        "iron": "Iron Sword",
        "axe": "Battle Axe",
        "battle": "Battle Axe",
    }
    OBJECT_ALIASES = {
        "treasure chest": "Treasure Chest",
        "chest": "Treasure Chest",
        "treasure": "Treasure Chest",
        "box": "Treasure Chest",
        "loot chest": "Treasure Chest",
        "locked chest": "Treasure Chest",
    }
    MONSTER_ALIASES = {
        "goblin": "Goblin",
        "monster": "Goblin",
        "enemy": "Goblin",
        "creature": "Goblin",
    }
    MOVE_DELTAS = {
        "north": (-1, 0),
        "south": (1, 0),
        "east": (0, 1),
        "west": (0, -1),
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.state = {
            "grid_size": [self.GRID_HEIGHT, self.GRID_WIDTH],
            "position": [6, 1],
            "hp": 70,
            "max_hp": 100,
            "weapon": "Rusty Dagger",
            "inventory": [],
            "map_items": self._generate_random_map(),
            "monster": {
                "name": self.MONSTER_NAME,
                "position": [5, 5],
                "hp": 36,
                "max_hp": 36,
                "alive": True,
            },
            "chest_opened": False,
            "victory": False,
            "game_over": False,
        }

    def _generate_random_map(self):
        """Generate random positions for map items, avoiding player start and monster."""
        occupied = {(6, 1), (5, 5)}  # Player start, Monster
        map_items = {"5,5": self.MONSTER_NAME}
        
        # Items to place: potion, Iron Sword, Battle Axe, Treasure Chest
        items_to_place = [self.POTION_NAME, "Iron Sword", "Battle Axe", self.CHEST_NAME]
        
        for item in items_to_place:
            # Find an unoccupied random position (but keep chest accessible)
            while True:
                y = random.randint(1, self.GRID_HEIGHT - 2)
                x = random.randint(1, self.GRID_WIDTH - 2)
                if (y, x) not in occupied:
                    occupied.add((y, x))
                    map_items[f"{y},{x}"] = item
                    break
        
        return map_items

    def fulfill_game_intent(self, intent, slots):
        if self.state.get("game_over") and intent != "restart_game":
            return self._response(
                "error",
                [
                    "You have fallen in the dungeon. Say restart game to try again.",
                    "The run is over. Restart the game if you want another attempt.",
                ],
            )

        if intent == "move":
            return self._move_player(slots.get("direction", ""))
        if intent == "inspect_room":
            return self._inspect_room()
        if intent == "pick_up":
            return self._pick_up_item(slots.get("item_name", ""))
        if intent == "use_item":
            return self._use_item(slots.get("item_name", ""))
        if intent == "attack":
            return self._attack_monster(slots.get("monster_name", ""))
        if intent == "open_object":
            return self._open_object(slots.get("object_name", ""))
        if intent == "equip_item":
            return self._equip_item(slots.get("item_name", ""))
        if intent == "show_inventory":
            return self._show_inventory()
        if intent == "show_status":
            return self._show_status()
        if intent == "restart_game":
            self.reset_game()
            return self._response(
                "success",
                [
                    "The dungeon shifts back into place. Game restarted.",
                    "Everything resets. The dungeon is ready again.",
                ],
            )

        return self._response("error", [f"Game intent '{intent}' not recognized."])

    def _move_player(self, direction):
        delta = self.MOVE_DELTAS.get(direction.lower())
        if delta is None:
            return self._response(
                "error",
                [
                    "I do not know which way that is.",
                    "That direction does not make sense to me.",
                ],
            )

        y, x = self.state["position"]
        new_y = y + delta[0]
        new_x = x + delta[1]
        if not (0 <= new_y < self.GRID_HEIGHT and 0 <= new_x < self.GRID_WIDTH):
            return self._response(
                "error",
                [
                    "A wall blocks that path.",
                    "You cannot move any farther in that direction.",
                ],
            )

        self.state["position"] = [new_y, new_x]
        current_item = self._current_tile_item()
        suffix = ""
        if current_item == self.MONSTER_NAME and self.state["monster"]["alive"]:
            suffix = " The Goblin is right in front of you."
        elif current_item == self.CHEST_NAME:
            suffix = " The Treasure Chest is here."
        elif current_item and self._is_pickupable(current_item):
            suffix = f" You spot a {current_item} here."

        return self._response(
            "success",
            [
                f"You moved {direction.title()}.{suffix}",
                f"You head {direction.title()} and stop at a new tile.{suffix}",
            ],
        )

    def _inspect_room(self):
        current_item = self._current_tile_item()
        parts = []

        if current_item == self.MONSTER_NAME and self.state["monster"]["alive"]:
            parts.append("The Goblin is on this tile and looks hostile.")
        elif current_item == self.CHEST_NAME:
            parts.append("The Treasure Chest is here.")
        elif current_item == self.OPEN_CHEST_NAME:
            parts.append("The treasure is already open.")
        elif current_item and self._is_pickupable(current_item):
            parts.append(f"You can pick up a {current_item} here.")
        else:
            parts.append("There is nothing on this tile.")

        if self.state["monster"]["alive"]:
            parts.append(
                f"The Goblin is {self._relative_direction(self.state['monster']['position'])} from you."
            )
        else:
            key_position = self._find_item_position(self.KEY_NAME)
            if key_position:
                parts.append(f"The Rusty Key is {self._relative_direction(key_position)} from you.")

        chest_position = self._find_item_position(self.CHEST_NAME) or self._find_item_position(self.OPEN_CHEST_NAME)
        if chest_position:
            if self.state["chest_opened"]:
                parts.append(f"The opened treasure is {self._relative_direction(chest_position)} from you.")
            else:
                parts.append(f"The Treasure Chest is {self._relative_direction(chest_position)} from you.")

        weapon_positions = []
        for weapon_name in self.WEAPON_ITEMS:
            pos = self._find_item_position(weapon_name)
            if pos:
                weapon_positions.append(f"{weapon_name} {self._relative_direction(pos)}")
        if weapon_positions:
            parts.append("Nearby gear: " + ", ".join(weapon_positions[:2]) + ".")

        return self._response(
            "success",
            [
                " ".join(parts),
                "You take stock of the room. " + " ".join(parts),
            ],
        )

    def _pick_up_item(self, requested_item):
        current_item = self._current_tile_item()
        if not current_item:
            return self._response(
                "error",
                [
                    "There is nothing here to pick up.",
                    "This tile is empty. There is nothing to collect.",
                ],
            )
        if not self._is_pickupable(current_item):
            return self._response(
                "error",
                [
                    f"You cannot pick up the {current_item}.",
                    f"The {current_item} is fixed in place.",
                ],
            )

        requested = self._canonical_item_name(requested_item)
        if requested and requested != current_item:
            # If the user likely meant the current tile item, accept it.
            if self._items_are_similar(requested, current_item):
                requested = current_item
            else:
                return self._response(
                    "error",
                    [
                        f"The {requested} is not on this tile.",
                        f"You do not see a {requested} here.",
                    ],
                )

        self.state["inventory"].append(current_item)
        del self.state["map_items"][self._position_key(self.state["position"])]

        if current_item in self.WEAPON_ITEMS:
            self.state["weapon"] = current_item
            return self._response(
                "success",
                [
                    f"You pick up the {current_item} and equip it immediately.",
                    f"The {current_item} is now in your hands.",
                ],
            )

        return self._response(
            "success",
            [
                f"You picked up the {current_item}.",
                f"You add the {current_item} to your inventory.",
            ],
        )

    def _use_item(self, requested_item):
        item = self._canonical_item_name(requested_item)
        if not item:
            if self.POTION_NAME in self.state["inventory"]:
                item = self.POTION_NAME
            else:
                return self._response(
                    "error",
                    [
                        "Tell me which item to use.",
                        "I need to know which item you want to use.",
                    ],
                )

        if item not in self.state["inventory"]:
            current_item = self._current_tile_item()
            if current_item and self._items_are_similar(item, current_item):
                return self._pick_up_item(current_item)
            return self._response(
                "error",
                [
                    f"You do not have a {item}.",
                    f"The {item} is not in your inventory.",
                ],
            )

        if item == self.POTION_NAME:
            self.state["inventory"].remove(item)
            heal_amount = 35
            previous_hp = self.state["hp"]
            self.state["hp"] = min(self.state["hp"] + heal_amount, self.state["max_hp"])
            healed = self.state["hp"] - previous_hp
            return self._response(
                "success",
                [
                    f"You drink the {item} and recover {healed} HP.",
                    f"The {item} restores you for {healed} HP.",
                ],
            )

        if item in self.WEAPON_ITEMS:
            self.state["weapon"] = item
            return self._response(
                "success",
                [
                    f"You equip the {item}.",
                    f"The {item} is now your active weapon.",
                ],
            )

        return self._response(
            "error",
            [
                f"You cannot use the {item} right now.",
                f"The {item} has no active use at the moment.",
            ],
        )

    def _attack_monster(self, requested_target):
        monster = self.state["monster"]
        if not monster["alive"]:
            return self._response(
                "error",
                [
                    "The Goblin is already defeated.",
                    "There is no living monster left to attack.",
                ],
            )

        target = self._canonical_monster_name(requested_target or self.MONSTER_NAME)
        if target != self.MONSTER_NAME:
            return self._response(
                "error",
                [
                    "That target is not in this dungeon.",
                    "You cannot attack that here.",
                ],
            )

        distance = self._distance(self.state["position"], monster["position"])
        if distance > 1:
            return self._response(
                "error",
                [
                    f"The Goblin is too far away. It is {self._relative_direction(monster['position'])} from you.",
                    "You need to get closer before you attack.",
                ],
            )

        damage_low, damage_high = self.WEAPON_DAMAGE.get(self.state["weapon"], (6, 10))
        player_damage = random.randint(damage_low, damage_high)
        monster["hp"] = max(0, monster["hp"] - player_damage)

        if monster["hp"] == 0:
            monster["alive"] = False
            self._remove_map_item(self.MONSTER_NAME)
            drop_position = self._drop_key()
            drop_hint = self._relative_direction(drop_position)
            return self._response(
                "success",
                [
                    f"You swing your {self.state['weapon']} at the Goblin for {player_damage} damage and defeat it. The Rusty Key drops {drop_hint}.",
                    f"The Goblin falls after your {self.state['weapon']} strike deals {player_damage} damage. The Rusty Key clatters {drop_hint}.",
                ],
            )

        retaliation = random.randint(6, 12)
        self.state["hp"] -= retaliation

        if self.state["hp"] <= 0 and self.POTION_NAME in self.state["inventory"]:
            self.state["inventory"].remove(self.POTION_NAME)
            self.state["hp"] = 50
            return self._response(
                "success",
                [
                    f"You strike with your {self.state['weapon']} for {player_damage} damage. The Goblin hits back for {retaliation}, but your Health Potion saves you automatically.",
                    f"Your {self.state['weapon']} deals {player_damage} damage. The Goblin's counterattack would have dropped you, so your Health Potion triggers and keeps you alive.",
                ],
            )

        if self.state["hp"] <= 0:
            self.state["hp"] = 0
            self.state["game_over"] = True
            return self._response(
                "error",
                [
                    f"You attack with your {self.state['weapon']} for {player_damage} damage, but the Goblin hits back for {retaliation} and defeats you.",
                    f"Your {self.state['weapon']} wounds the Goblin for {player_damage}, but you fall to its {retaliation} damage counterattack.",
                ],
            )

        return self._response(
            "success",
            [
                f"You swing your {self.state['weapon']} at the Goblin for {player_damage} damage. It hits back for {retaliation}. The Goblin has {monster['hp']} HP left.",
                f"Your {self.state['weapon']} deals {player_damage} damage, and the Goblin answers with {retaliation}. It has {monster['hp']} HP remaining.",
            ],
        )

    def _open_object(self, requested_object):
        target_object = self._canonical_object_name(requested_object or self.CHEST_NAME)
        if target_object != self.CHEST_NAME:
            return self._response(
                "error",
                [
                    "That object cannot be opened here.",
                    "There is nothing like that to unlock in this room.",
                ],
            )

        chest_position = self._find_item_position(self.CHEST_NAME) or self._find_item_position(self.OPEN_CHEST_NAME)
        if chest_position != self.state["position"]:
            if chest_position:
                return self._response(
                    "error",
                    [
                        f"The Treasure Chest is not on this tile. It is {self._relative_direction(chest_position)} from you.",
                        "You need to stand on the chest tile before opening it.",
                    ],
                )
            return self._response(
                "error",
                [
                    "The Treasure Chest is nowhere to be found.",
                    "There is no chest left to open.",
                ],
            )

        if self.state["chest_opened"]:
            return self._response(
                "error",
                [
                    "The Treasure Chest is already open.",
                    "You already unlocked the treasure.",
                ],
            )

        if self.KEY_NAME not in self.state["inventory"]:
            return self._response(
                "error",
                [
                    "The Treasure Chest is locked. You still need the Rusty Key.",
                    "The lock will not budge without the Rusty Key.",
                ],
            )

        self.state["inventory"].remove(self.KEY_NAME)
        self.state["chest_opened"] = True
        self.state["victory"] = True
        self.state["map_items"][self._position_key(chest_position)] = self.OPEN_CHEST_NAME
        return self._response(
            "success",
            [
                "You unlock the Treasure Chest and claim the treasure. You win.",
                "The Rusty Key turns, the chest opens, and the treasure is yours.",
            ],
        )

    def _equip_item(self, requested_item):
        item = self._canonical_item_name(requested_item)
        if not item:
            return self._response(
                "error",
                [
                    "Tell me which weapon to equip.",
                    "I need to know which weapon you want to equip.",
                ],
            )

        if item not in self.state["inventory"]:
            return self._response(
                "error",
                [
                    f"You do not have a {item} in your inventory.",
                    f"The {item} is not available to equip.",
                ],
            )

        if item not in self.WEAPON_ITEMS:
            return self._response(
                "error",
                [
                    f"The {item} is not a weapon.",
                    f"You cannot equip the {item}.",
                ],
            )

        self.state["weapon"] = item
        return self._response(
            "success",
            [
                f"You draw the {item}. It's now your active weapon.",
                f"The {item} is now equipped.",
            ],
        )

    def _show_inventory(self):
        if not self.state["inventory"]:
            return self._response(
                "success",
                [
                    "Your inventory is empty.",
                    "You are not carrying anything.",
                ],
            )

        items = self.state["inventory"]
        item_list = ", ".join(items[:-1]) + f", and {items[-1]}" if len(items) > 1 else items[0]
        return self._response(
            "success",
            [
                f"You are carrying: {item_list}.",
                f"Your inventory contains: {item_list}.",
            ],
        )

    def _show_status(self):
        hp = self.state["hp"]
        max_hp = self.state["max_hp"]
        weapon = self.state["weapon"]
        position = self.state["position"]
        monster_status = ""
        
        if self.state["monster"]["alive"]:
            monster_hp = self.state["monster"]["hp"]
            monster_max_hp = self.state["monster"]["max_hp"]
            monster_status = f" The Goblin is at {monster_hp} out of {monster_max_hp} hit points."
        else:
            monster_status = " The Goblin is defeated."

        chest_status = " The Treasure Chest is already opened." if self.state["chest_opened"] else " The Treasure Chest awaits on the eastern side of the dungeon."

        return self._response(
            "success",
            [
                f"You have {hp} out of {max_hp} hit points. You are wielding the {weapon}. Your position is {position}.{monster_status}{chest_status}",
                f"Health: {hp}/{max_hp}. Weapon: {weapon}. Position: {position}.{monster_status}{chest_status}",
            ],
        )

    def _drop_key(self):
        occupied = {
            tuple(self.state["position"]),
            tuple(self.state["monster"]["position"]),
        }
        for pos_str, item_name in self.state["map_items"].items():
            y, x = map(int, pos_str.split(","))
            if item_name != self.MONSTER_NAME:
                occupied.add((y, x))

        candidates = [
            (y, x)
            for y in range(1, self.GRID_HEIGHT - 1)
            for x in range(1, self.GRID_WIDTH - 1)
            if (y, x) not in occupied
        ]
        drop_y, drop_x = random.choice(candidates)
        self.state["map_items"][f"{drop_y},{drop_x}"] = self.KEY_NAME
        return [drop_y, drop_x]

    def _current_tile_item(self):
        return self.state["map_items"].get(self._position_key(self.state["position"]))

    def _find_item_position(self, item_name):
        for pos_str, current_item in self.state["map_items"].items():
            if current_item == item_name:
                y, x = map(int, pos_str.split(","))
                return [y, x]
        return None

    def _remove_map_item(self, item_name):
        for pos_str, current_item in list(self.state["map_items"].items()):
            if current_item == item_name:
                del self.state["map_items"][pos_str]
                return

    def _is_pickupable(self, item_name):
        return item_name not in {self.MONSTER_NAME, self.CHEST_NAME, self.OPEN_CHEST_NAME}

    def _canonical_item_name(self, name):
        if not name:
            return ""
        lowered = name.lower().strip()
        if lowered in self.ITEM_ALIASES:
            return self.ITEM_ALIASES[lowered]

        words = set(re.findall(r"[a-z]+", lowered))
        for keyword, canonical in self.ITEM_KEYWORD_ALIASES.items():
            if keyword in words:
                return canonical

        return name.title()

    @staticmethod
    def _items_are_similar(requested, current_item):
        requested_words = set(re.findall(r"[a-z]+", requested.lower()))
        current_words = set(re.findall(r"[a-z]+", current_item.lower()))
        return bool(requested_words & current_words)

    def _canonical_object_name(self, name):
        if not name:
            return ""
        lowered = name.lower().strip()
        return self.OBJECT_ALIASES.get(lowered, name.title())

    def _canonical_monster_name(self, name):
        if not name:
            return ""
        lowered = name.lower().strip()
        words = set(re.findall(r"[a-z]+", lowered))
        for alias, canonical in self.MONSTER_ALIASES.items():
            if alias in words or alias == lowered:
                return canonical
        return self.MONSTER_ALIASES.get(lowered, name.title())

    @staticmethod
    def _position_key(position):
        return f"{position[0]},{position[1]}"

    @staticmethod
    def _distance(position_a, position_b):
        return abs(position_a[0] - position_b[0]) + abs(position_a[1] - position_b[1])

    def _relative_direction(self, target_position):
        y, x = self.state["position"]
        target_y, target_x = target_position
        vertical = ""
        horizontal = ""
        if target_y < y:
            vertical = "north"
        elif target_y > y:
            vertical = "south"
        if target_x < x:
            horizontal = "west"
        elif target_x > x:
            horizontal = "east"

        if vertical and horizontal:
            return f"to the {vertical} {horizontal}"
        if vertical:
            return f"to the {vertical}"
        if horizontal:
            return f"to the {horizontal}"
        return "right here"

    def _response(self, status, templates):
        return {
            "status": status,
            "message": random.choice(templates),
            "state": self.state,
        }
