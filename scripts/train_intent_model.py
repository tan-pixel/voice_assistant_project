import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Configurations
MODEL_DIR = "../data/models"
# Keep generated model artifacts aligned with intents the app actually routes.
EXCLUDED_INTENTS = set()

# Define the Dataset for D&D, Game, and Basic Intents
# Using the BIO inline annotation format
intent_data = {

    "lookup_monster": [
        "tell me about goblins/B-monster_name",
        "what is a dragon/B-monster_name",
        "look up stats for a zombie/B-monster_name",
        "give me info on a skeleton/B-monster_name",
        "what are the abilities of an orc/B-monster_name",
        "show me details for a vampire/B-monster_name",
        "describe a troll/B-monster_name",
        "how strong is a werewolf/B-monster_name",
        "what does a lich/B-monster_name do",
        "stats for a demon/B-monster_name",
        "tell me weaknesses of a ghost/B-monster_name",
        "information about a hydra/B-monster_name",
        "what are the stats of a basilisk/B-monster_name",
        "explain a minotaur/B-monster_name",
        "how dangerous is a harpy/B-monster_name",
        "give me the profile of a golem/B-monster_name",
        "what powers does a necromancer/B-monster_name have",
        "describe a giant/B-monster_name",
        "what kind of creature is a wraith/B-monster_name",
        "show attributes of a centaur/B-monster_name",
        "what is a slime/B-monster_name",
        "details for a phoenix/B-monster_name"
    ],

    "lookup_spell": [
        "what does fireball/B-spell_name do",
        "how much damage does magic/B-spell_name missile/I-spell_name deal",
        "explain the shield/B-spell_name spell",
        "tell me about lightning/B-spell_name bolt/I-spell_name",
        "what is the effect of invisibility/B-spell_name",
        "describe the heal/B-spell_name spell",
        "how does frost/B-spell_name nova/I-spell_name work",
        "what does teleport/B-spell_name do",
        "give me details on summon/B-spell_name familiar/I-spell_name",
        "what is charm/B-spell_name person/I-spell_name",
        "how strong is meteor/B-spell_name strike/I-spell_name",
        "explain poison/B-spell_name cloud/I-spell_name",
        "what does arcane/B-spell_name blast/I-spell_name do",
        "describe the slow/B-spell_name spell",
        "what is the range of blink/B-spell_name",
        "tell me how drain/B-spell_name life/I-spell_name works",
        "what does silence/B-spell_name do",
        "explain summon/B-spell_name elemental/I-spell_name",
        "how effective is ice/B-spell_name shard/I-spell_name",
        "what does barrier/B-spell_name do",
        "give details on curse/B-spell_name",
        "what is the purpose of haste/B-spell_name"
    ],

    "equipment_category": [
        "show me martial/B-category weapons/I-category",
        "list all simple/B-category weapons/I-category",
        "what is in the armor/B-category category",
        "display ranged/B-category weapons/I-category",
        "what items are in heavy/B-category armor/I-category",
        "show all light/B-category armor/I-category",
        "list magic/B-category items/I-category",
        "what weapons are in exotic/B-category weapons/I-category",
        "give me all shields/B-category",
        "what gear is in potions/B-category",
        "show available rings/B-category",
        "list all equipment in tools/B-category",
        "show me all melee/B-category weapons/I-category",
        "list defensive/B-category gear/I-category",
        "what items belong to accessories/B-category",
        "display all helmets/B-category",
        "show boots/B-category category",
        "list all consumables/B-category",
        "what is inside scrolls/B-category",
        "show available amulets/B-category",
        "list crafting/B-category materials/I-category",
        "what equipment is under artifacts/B-category"
    ],

    "lookup_weapon": [
        "tell me about the longsword/B-weapon_name",
        "what does a dagger/B-weapon_name do",
        "show me the stats for a battleaxe/B-weapon_name",
        "how strong is the greatsword/B-weapon_name",
        "describe the shortbow/B-weapon_name weapon",
        "what are the properties of a light/B-weapon_name crossbow/I-weapon_name",
        "give me info on the handaxe/B-weapon_name",
        "how much damage does a warhammer/B-weapon_name deal",
        "look up the rapier/B-weapon_name",
        "tell me about the heavy/B-weapon_name crossbow/I-weapon_name",
        "show details for the mace/B-weapon_name",
        "what kind of weapon is the spear/B-weapon_name",
        "give me the profile of a javelin/B-weapon_name",
        "explain the trident/B-weapon_name",
        "look up the morningstar/B-weapon_name weapon",
        "what are the stats of a longbow/B-weapon_name",
        "describe the glaive/B-weapon_name",
        "tell me about the maul/B-weapon_name",
        "how effective is the quarterstaff/B-weapon_name",
        "show me the properties of the whip/B-weapon_name"
    ],

    "lookup_armor": [
        "tell me about leather/B-armor_name armor/I-armor_name",
        "what does chain/B-armor_name mail/I-armor_name do",
        "show me the stats for plate/B-armor_name armor/I-armor_name",
        "describe studded/B-armor_name leather/I-armor_name armor/I-armor_name",
        "look up the shield/B-armor_name",
        "give me info on half/B-armor_name plate/I-armor_name",
        "how good is splint/B-armor_name armor/I-armor_name",
        "what are the properties of padded/B-armor_name armor/I-armor_name",
        "tell me about ring/B-armor_name mail/I-armor_name",
        "show details for scale/B-armor_name mail/I-armor_name",
        "what does breastplate/B-armor_name do",
        "give me the stats for hide/B-armor_name armor/I-armor_name",
        "describe chain/B-armor_name shirt/I-armor_name",
        "look up spiked/B-armor_name armor/I-armor_name",
        "tell me about the shield/B-armor_name armor/I-armor_name",
        "what kind of armor is padded/B-armor_name armor/I-armor_name",
        "show me the profile of plate/B-armor_name",
        "how protective is leather/B-armor_name armor/I-armor_name",
        "explain scale/B-armor_name mail/I-armor_name armor/I-armor_name",
        "what are the details of breastplate/B-armor_name armor/I-armor_name"
    ],

    "lookup_class": [
        "tell me about the fighter/B-class_name class/I-class_name",
        "what does the wizard/B-class_name class/I-class_name do",
        "show me the barbarian/B-class_name class/I-class_name",
        "give me info on the rogue/B-class_name class/I-class_name",
        "how does the cleric/B-class_name class/I-class_name work",
        "describe the ranger/B-class_name class/I-class_name",
        "what are the features of the monk/B-class_name class/I-class_name",
        "look up the paladin/B-class_name class/I-class_name",
        "tell me about the druid/B-class_name class/I-class_name",
        "show me the bard/B-class_name class/I-class_name",
        "explain the sorcerer/B-class_name class/I-class_name",
        "what are the basics of the warlock/B-class_name class/I-class_name",
        "give me the profile of the barbarian/B-class_name class/I-class_name",
        "how strong is the fighter/B-class_name class/I-class_name",
        "what does the rogue/B-class_name class/I-class_name specialize in",
        "show details for the wizard/B-class_name class/I-class_name",
        "tell me how the paladin/B-class_name class/I-class_name works",
        "describe the bard/B-class_name",
        "look up the monk/B-class_name",
        "what are the proficiencies of the ranger/B-class_name class/I-class_name"
    ],

    "lookup_race": [
        "tell me about the elf/B-race_name race/I-race_name",
        "what does the dwarf/B-race_name race/I-race_name get",
        "show me the human/B-race_name race/I-race_name",
        "describe the half/B-race_name elf/I-race_name race/I-race_name",
        "give me info on the dragonborn/B-race_name race/I-race_name",
        "how does the tiefling/B-race_name race/I-race_name work",
        "look up the half/B-race_name orc/I-race_name race/I-race_name",
        "tell me about the halfling/B-race_name race/I-race_name",
        "what are the traits of the gnome/B-race_name race/I-race_name",
        "show details for the human/B-race_name race/I-race_name",
        "explain the dragonborn/B-race_name",
        "what are the features of the elf/B-race_name race/I-race_name",
        "give me the profile of the dwarf/B-race_name race/I-race_name",
        "look up the tiefling/B-race_name",
        "describe the halfling/B-race_name",
        "tell me how the gnome/B-race_name race/I-race_name works",
        "what does the half/B-race_name orc/I-race_name race/I-race_name get",
        "show me the traits of the half/B-race_name elf/I-race_name",
        "give details on the human/B-race_name",
        "what should i know about the dragonborn/B-race_name race/I-race_name"
    ],

    "lookup_condition": [
        "tell me about the blinded/B-condition_name condition/I-condition_name",
        "what does charmed/B-condition_name do",
        "show me the stunned/B-condition_name condition/I-condition_name",
        "describe the poisoned/B-condition_name condition/I-condition_name",
        "how does restrained/B-condition_name work",
        "give me info on the prone/B-condition_name condition/I-condition_name",
        "look up the invisible/B-condition_name condition/I-condition_name",
        "what happens when someone is grappled/B-condition_name",
        "tell me about petrified/B-condition_name",
        "show details for unconscious/B-condition_name",
        "explain the frightened/B-condition_name condition/I-condition_name",
        "what does deafened/B-condition_name do",
        "describe the paralyzed/B-condition_name condition/I-condition_name",
        "give me the rules for exhaustion/B-condition_name",
        "how does incapacitated/B-condition_name work",
        "look up the poisoned/B-condition_name effect/I-condition_name",
        "what happens when a creature is blinded/B-condition_name",
        "tell me how charmed/B-condition_name works",
        "show me the details of restrained/B-condition_name",
        "what should i know about prone/B-condition_name"
    ],

    "move": [
        "move north/B-direction",
        "go south/B-direction",
        "walk east/B-direction",
        "head west/B-direction",
        "run north/B-direction",
        "travel south/B-direction",
        "step east/B-direction",
        "move towards west/B-direction",
        "go up/B-direction",
        "go down/B-direction",
        "head north/B-direction quickly",
        "walk south/B-direction slowly",
        "proceed east/B-direction",
        "advance north/B-direction",
        "continue south/B-direction",
        "go straight east/B-direction",
        "turn west/B-direction",
        "move north/B-direction now",
        "quickly go south/B-direction",
        "slowly head east/B-direction",
        "proceed to the north/B-direction",
        "walk toward south/B-direction",
        "move east/B-direction immediately",
        "north/B-direction",
        "south/B-direction",
        "east/B-direction",
        "west/B-direction",
        "up/B-direction",
        "down/B-direction"
    ],

    "attack": [
        "attack the goblin/B-monster_name",
        "hit the goblin/B-monster_name",
        "kill the goblin/B-monster_name",
        "strike the monster/B-monster_name",
        "fight the goblin/B-monster_name",
        "slash the enemy/B-monster_name",
        "stab the goblin/B-monster_name",
        "take down the monster/B-monster_name",
        "finish off the goblin/B-monster_name",
        "deal damage to the monster/B-monster_name",
        "attack that goblin/B-monster_name now",
        "swing at the enemy/B-monster_name",
        "smash the monster/B-monster_name",
        "fight the enemy/B-monster_name with my weapon",
        "go attack the goblin/B-monster_name",
        "hit the monster/B-monster_name hard",
        "strike the goblin/B-monster_name again",
        "attack the enemy/B-monster_name with the axe",
        "slay the goblin/B-monster_name",
        "stab the monster/B-monster_name now",
        "fight the goblin/B-monster_name right away",
        "attack the monster/B-monster_name in front of me",
        "attack it/B-monster_name",
        "hit it/B-monster_name",
        "fight it/B-monster_name"
    ],

    "inspect_room": [
        "look around",
        "inspect the room",
        "search the room",
        "what is in here",
        "what do i see",
        "scan the area",
        "check the room",
        "show me the room",
        "examine the room",
        "look around the dungeon",
        "tell me what is nearby",
        "what is around me",
        "inspect my surroundings",
        "search this area",
        "do a room check",
        "take a look around",
        "show nearby objects",
        "what is here",
        "give me the room details",
        "examine my surroundings",
        "scan this room",
        "tell me what is in the area"
    ],

    "open_object": [
        "open the chest/B-object_name",
        "unlock the chest/B-object_name",
        "open the treasure/B-object_name chest/I-object_name",
        "unlock the treasure/B-object_name chest/I-object_name",
        "open the treasure/B-object_name",
        "use the key on the chest/B-object_name",
        "open that chest/B-object_name",
        "unlock that chest/B-object_name",
        "open the locked chest/B-object_name",
        "unlock the locked chest/B-object_name",
        "open the treasure/B-object_name box/I-object_name",
        "use the rusty key on the chest/B-object_name",
        "can i open the chest/B-object_name",
        "let me open the treasure/B-object_name chest/I-object_name",
        "try to open the chest/B-object_name",
        "open up the chest/B-object_name",
        "unlock the treasure/B-object_name",
        "open the loot/B-object_name chest/I-object_name",
        "use my key on the treasure/B-object_name chest/I-object_name",
        "open the prize/B-object_name chest/I-object_name",
        "unlock this chest/B-object_name",
        "open the big chest/B-object_name"
    ],

    "pick_up": [
        "pick up the sword/B-item_name",
        "grab the key/B-item_name",
        "take the health/B-item_name potion/I-item_name",
        "collect the shield/B-item_name",
        "pick up gold/B-item_name coins/I-item_name",
        "grab the magic/B-item_name scroll/I-item_name",
        "take the dagger/B-item_name",
        "loot the chest/B-item_name",
        "pick up the armor/B-item_name",
        "grab a mana/B-item_name potion/I-item_name",
        "take the ring/B-item_name",
        "collect the treasure/B-item_name",
        "pick up the helmet/B-item_name",
        "grab the boots/B-item_name",
        "take the staff/B-item_name",
        "collect the gem/B-item_name",
        "pick up arrows/B-item_name",
        "grab the bow/B-item_name",
        "take the potion/B-item_name",
        "collect coins/B-item_name",
        "pick up the artifact/B-item_name",
        "grab the loot/B-item_name",
        "take potion/B-item_name",
        "pick up potion/B-item_name",
        "grab potion/B-item_name",
        "take the key/B-item_name"
    ],

    "use_item": [
        "use a potion/B-item_name",
        "drink the health/B-item_name potion/I-item_name",
        "drink the potion/B-item_name",
        "use the potion/B-item_name",
        "use the key/B-item_name",
        "consume a mana/B-item_name potion/I-item_name",
        "consume the potion/B-item_name",
        "take the potion/B-item_name",
        "activate the scroll/B-item_name",
        "equip the sword/B-item_name",
        "wear the armor/B-item_name",
        "use the shield/B-item_name",
        "drink the elixir/B-item_name",
        "apply the bandage/B-item_name",
        "use the magic/B-item_name ring/I-item_name",
        "equip the helmet/B-item_name",
        "use the boots/B-item_name",
        "activate the artifact/B-item_name",
        "apply the potion/B-item_name",
        "activate the amulet/B-item_name",
        "use the scroll/B-item_name now",
        "drink potion/B-item_name",
        "use potion/B-item_name",
        "take potion/B-item_name",
        "quaff the potion/B-item_name"
    ],

    "equip_item": [
        "equip the sword/B-item_name",
        "switch to the axe/B-item_name",
        "equip the dagger/B-item_name",
        "wield the sword/B-item_name",
        "draw the battle/B-item_name axe/I-item_name",
        "equip the iron/B-item_name sword/I-item_name",
        "switch weapons to the axe/B-item_name",
        "take the sword/B-item_name",
        "put on the armor/B-item_name",
        "equip my weapon/B-item_name",
        "switch to my sword/B-item_name",
        "equip the battle/B-item_name axe/I-item_name",
        "draw the dagger/B-item_name",
        "wield the axe/B-item_name",
        "switch my gear to the sword/B-item_name",
        "equip the larger weapon/B-item_name",
        "change weapon to the axe/B-item_name",
        "ready the sword/B-item_name",
        "take up the dagger/B-item_name",
        "outfit myself with the sword/B-item_name"
    ],

    "show_inventory": [
        "show my inventory",
        "what am i carrying",
        "list my items",
        "show my items",
        "what is in my inventory",
        "check my inventory",
        "show what i have",
        "what am i holding",
        "list my gear",
        "display my inventory",
        "what items do i have",
        "show me what i am carrying",
        "tell me my inventory",
        "what do i have with me",
        "show my current items",
        "list everything i am carrying",
        "check what i have",
        "display my items",
        "show all my gear",
        "what is in my pack"
    ],

    "show_status": [
        "show my status",
        "what is my status",
        "show my health",
        "tell me my stats",
        "what is my current health",
        "show my current status",
        "what is my hp",
        "tell me my position",
        "show my weapon",
        "give me my status",
        "what is the dungeon status",
        "tell me where i am",
        "show my condition",
        "check my status",
        "display my stats",
        "what weapon am i holding",
        "tell me my health points",
        "show my current position",
        "what is my situation",
        "give me a status report"
    ],

    "restart_game": [
        "restart the game",
        "start over",
        "reset the dungeon",
        "restart everything",
        "begin a new game",
        "reset the game",
        "start a new run",
        "wipe progress and restart",
        "reboot the game",
        "restart from the beginning",
        "reset all progress",
        "start fresh",
        "restart the whole game",
        "start the game again",
        "reset my progress",
        "clear everything and restart",
        "begin from scratch",
        "restart the level",
        "reload the game from start",
        "reset the adventure",
        "restart the session",
        "start the dungeon again"
    ],

    "Greetings": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "hello there",
        "hi assistant",
        "hey there",
        "good afternoon",
        "hello assistant",
        "hi there",
        "hey assistant",
        "good day",
        "hello again",
        "hi again",

        # Basic & Informal
        "Hey!",
        "Hi there.",
        "Howdy!",
        "Yo.",
        "Hi hi.",

        # Time-based
        "Good morning, how are you?",
        "Good afternoon, assistant.",
        "A very good evening to you.",
        "Good day.",
        "Morning!",

        # Conversational/Check-ins
        "How's it going today?",
        "What's up?",
        "How's everything?",
        "How are things?",
        "Nice to see you.",
        "Hey, how've you been?",
        "What's new?",
        "How are you doing this morning?",

        # Formal/Polite
        "Greetings.",
        "Hello, I hope you're having a good day.",
        "Hello, it's nice to speak with you.",
        "I'd like to say hello.",
        "Pleasure to meet you.",

        # Returns/Again
        "Hello again.",
        "Hi, I'm back.",
        "Hey there, me again.",
        "Good to see you again.",

        # Compound Greetings
        "Hey, quick question for you.",
        "Hi assistant, can you help me?",
        "Hello, are you there?",
        "Hi, I need a hand with something."
    ],
    "Goodbye": [
        "bye",
        "goodbye",
        "see you",
        "talk to you later",
        "bye bye",
        "see you later",
        "good night",
        "catch you later",
        "farewell",
        "see you soon",
        "bye for now",
        "talk later",
        "have a good night",
        "goodbye assistant",
        "see you next time",

        # Short & Informal
        "Bye!",
        "Peace.",
        "Later.",
        "Bye bye.",
        "I'm out.",
        "Cya.",

        # Time-based Goodbyes
        "Good night.",
        "Have a great evening.",
        "Enjoy the rest of your day.",
        "Have a good weekend.",
        "Have a nice afternoon.",
        "Sleep well.",

        # Future-focused (See you...)
        "See you later.",
        "Catch you on the flip side.",
        "Talk to you soon.",
        "See ya in a bit.",
        "Talk to you next time.",
        "I'll catch you later.",
        "Until next time.",

        # Formal/Polite
        "Goodbye.",
        "Farewell.",
        "Take care.",
        "I'm heading out now, goodbye.",
        "I'm finished for today, thank you.",
        "It was nice talking to you.",
        "I'll be going now.",

        # Conversational "Sign-offs"
        "That's all I needed, thanks!",
        "I'm done here, bye.",
        "Okay, talk to you later then.",
        "Alright, see you.",
        "Thanks for the help, goodbye.",
        "I've got to go now, see ya."
    ],

    "OOS": [
        "sing a song for me",
        "How do I implement multithreading in Go?",
        "open my email",
        "tell me a joke",
        "who is the president",
        "translate hello to french",
        "turn on the lights",
        "book a flight",
        "what time is it",
        "search for restaurants",
        "play a movie",
        "open youtube",
        "increase volume",
        "show my calendar",
        "what is machine learning",

        # Media & Entertainment
        "Tell me a fun fact about space.",
        "Can you put on a movie?",
        "Skip this song.",
        "Turn the volume up a bit.",
        "What's playing on the radio right now?",
        "Open YouTube and search for cat videos.",

        # Information
        "Explain the difference between Intel and AMD chips.",
        "When is the next bus heading to the University of Ottawa?",
        "Who won the Super Bowl last year?",
        "Tell me a fun fact about space.",
        "How do you spell 'exaggerate'?",
        "What is the capital of France?",
        "Search for the best pizza places nearby.",

        # Productivity & Apps
        "Check my emails from this morning.",
        "What's on my calendar for Tuesday?",
        "Open the calculator app.",
        "Send a text to Mom saying I'm running late.",
        "Write a new note called 'Grocery List'.",
        "Can you book a flight to New York?",

        # Smart Home (Non-timer/alarm)
        "Turn off the kitchen lights.",
        "Lock the front door.",
        "Set the thermostat to 72 degrees.",
        "Is the garage door closed?",

        # Translation & Language
        "How do you say 'where is the bathroom' in Spanish?",
        "Translate 'good morning' to Japanese.",

        # Complex/Abstract Questions
        "What is the meaning of life?",
        "Can you explain how machine learning works?",
        "Tell me a joke to make me laugh.",
        "What is the current price of Bitcoin?",
        "Who is the current President of the United States?",
        "How many miles are in a kilometer?",
        "Find me a recipe for chocolate cake."
    ],

    "Help": [
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
        "what commands can i use"
    ],

    "Weather": [
        "what is the weather in Paris/B-city",
        "check the forecast for London/B-city",
        "tell me the weather in Tokyo/B-city",
        "how hot is it in Miami/B-city right now",
        "what is it like outside in Ottawa/B-city",
        "show me todays weather for Berlin/B-city",
        "give me the forecast for Madrid/B-city",
        "what is the temperature in Seattle/B-city",
        "how is the weather in Vancouver/B-city today",
        "can you check the weather for Dublin/B-city",
        "i want the weather report for Boston/B-city",
        "what will the weather be in Chicago/B-city",
        "tell me about the weather in Rome/B-city",
        "how warm is it in Lisbon/B-city",
        "what is going on weather wise in Cairo/B-city",
        "give me a weather update for Athens/B-city",
        "what is the forecast like in Prague/B-city",
        "can you look up the weather in Seoul/B-city",
        "what is the current weather in Dubai/B-city",
        "how does it feel outside in Delhi/B-city",
        "tell me the temperature in Phoenix/B-city",
        "check the weather for Denver/B-city",
        "i need the forecast for Halifax/B-city",
        "what is the weather report in Zurich/B-city",
        "can you tell me the weather in Moscow/B-city",
        "what is it like in Sydney/B-city today",
        "give me the latest weather for Nairobi/B-city",
        "how is the weather looking in Lagos/B-city",
        "what is the outlook for Havana/B-city",
        "show me the weather conditions in Manila/B-city",
        "what is the outlook for Havana/B-city",
        "show me the weather conditions in Manila/B-city",
        "tell me the weather in California/B-location right/B-time now/I-time",
        "check the forecast for Alaska/B-location tomorrow/B-time",
        "what is the weather like in Florida/B-location this/B-time evening/I-time",
        "give me the weather in Ontario/B-location next/B-time week/I-time",
        "show me the forecast for Texas/B-location this/B-time weekend/I-time",
        "tell me how it looks in Quebec/B-location tonight/B-time"
    ],

    "WeatherWind": [
        "what is the wind speed in Ottawa/B-city",
        "is it windy in Berlin/B-city",
        "tell me the wind conditions in Calgary/B-city",
        "how strong is the wind in Halifax/B-city",
        "check the breeze in Dublin/B-city",
        "is there a strong wind in Boston/B-city",
        "what are the gusts like in Lisbon/B-city",
        "give me the wind forecast for Denver/B-city",
        "is it breezy in Seattle/B-city today",
        "how hard is the wind blowing in Phoenix/B-city",
        "tell me if it is gusty in Reykjavik/B-city",
        "what is the wind doing in Rome/B-city",
        "do i have strong gusts in Madrid/B-city",
        "check if it is windy in Vancouver/B-city",
        "what is the breeze like in Tokyo/B-city",
        "is there much wind in Paris/B-city right now",
        "give me the wind report for Seoul/B-city",
        "how windy will it be in Dubai/B-city",
        "tell me the current wind speed in Athens/B-city",
        "is it blustery in Prague/B-city",
        "what are the gust conditions in Cairo/B-city",
        "check the wind for Zurich/B-city",
        "will it be breezy in Sydney/B-city",
        "how windy is it around Delhi/B-city",
        "tell me if there are gusts in Nairobi/B-city",
        "what is the wind speed in Lagos/B-city",
        "is it breezy in Havana/B-city",
        "check for wind in Manila/B-city",
        "do we have strong breeze in Moscow/B-city",
         "check for wind in Manila/B-city",
        "do we have strong breeze in Moscow/B-city",
        "how windy is it in California/B-location right/B-time now/I-time",
        "tell me the wind forecast for Alaska/B-location tomorrow/B-time",
        "is it going to be breezy in Florida/B-location this/B-time afternoon/I-time",
        "check the gusts in Ontario/B-location tonight/B-time",
        "what will the wind be like in Texas/B-location this/B-time weekend/I-time",
        "give me the breeze report for Quebec/B-location next/B-time week/I-time"
    ],

    "WeatherCondition": [
        "is it sunny in Miami/B-city",
        "is it cloudy in Seattle/B-city",
        "tell me the current conditions in Paris/B-city",
        "is it foggy in London/B-city",
        "what does the sky look like in Rome/B-city",
        "are conditions clear in Tokyo/B-city",
        "is it overcast in Berlin/B-city",
        "tell me the sky conditions in Ottawa/B-city",
        "is it stormy in Dublin/B-city",
        "what are conditions like in Boston/B-city",
        "check if it is clear in Lisbon/B-city",
        "is it gloomy in Prague/B-city today",
        "tell me whether it is sunny in Cairo/B-city",
        "are there clouds in Athens/B-city right now",
        "what is the weather condition in Seoul/B-city",
        "is it foggy in Dubai/B-city",
        "check if the skies are clear in Delhi/B-city",
        "tell me if it is cloudy in Phoenix/B-city",
        "what are the current conditions in Denver/B-city",
        "is it overcast in Halifax/B-city",
        "show me the sky conditions in Zurich/B-city",
        "is it bright and sunny in Sydney/B-city",
        "tell me if it is gloomy in Nairobi/B-city",
        "what does the sky look like in Lagos/B-city",
        "is it cloudy over Havana/B-city",
        "check whether it is clear in Manila/B-city",
        "tell me the conditions in Moscow/B-city",
        "is it stormy in Vancouver/B-city today",
        "what are conditions like in Calgary/B-city",
        "is it stormy in Vancouver/B-city today",
        "what are conditions like in Calgary/B-city",
        "what are the conditions in California/B-location right/B-time now/I-time",
        "tell me if the skies are clear in Alaska/B-location tomorrow/B-time",
        "is it cloudy in Florida/B-location this/B-time morning/I-time",
        "what will conditions be like in Ontario/B-location tonight/B-time",
        "check the sky conditions for Texas/B-location this/B-time weekend/I-time",
        "is it sunny in Quebec/B-location next/B-time week/I-time"
    ],

    "WeatherRain": [
       "will i need an umbrella in Dublin/B-city",
        "does it rain in Boston/B-city right now",
        "is there rain in Seattle/B-city today",
        "tell me if it is drizzling in London/B-city",
        "will it be rainy in Paris/B-city",
        "check for showers in Rome/B-city",
        "is there any rain in Ottawa/B-city",
        "am i going to get rain in Chicago/B-city",
        "are there showers in Berlin/B-city today",
        "is it pouring in Miami/B-city",
        "tell me if i need an umbrella in Lisbon/B-city",
        "will there be rain in Calgary/B-city",
        "is it a rainy day in Halifax/B-city",
        "check whether it is raining in Cairo/B-city",
        "is it drizzling in Tokyo/B-city",
        "will i need an umbrella in Seoul/B-city",
        "are there showers in Dubai/B-city",
        "is it raining in Delhi/B-city now",
        "check for rain in Phoenix/B-city",
        "will it be wet in Denver/B-city",
        "does Zurich/B-city have rain today",
        "is there a chance of showers in Sydney/B-city",
        "tell me if it is pouring in Nairobi/B-city",
        "is it rainy in Lagos/B-city",
        "will Havana/B-city get rain",
        "check if it is drizzling in Manila/B-city",
        "does Moscow/B-city need umbrellas today",
        "is there rain coming to Prague/B-city",
        "will Boston/B-city stay dry or rain",
        "is there rain coming to Prague/B-city",
        "will Boston/B-city stay dry or rain",
        "will it rain in California/B-location right/B-time now/I-time",
        "am i going to need an umbrella in Alaska/B-location tomorrow/B-time",
        "is there rain coming to Florida/B-location this/B-time evening/I-time",
        "check for showers in Ontario/B-location tonight/B-time",
        "will Texas/B-location get rain this/B-time weekend/I-time",
        "is it going to drizzle in Quebec/B-location next/B-time week/I-time"
    ],

    "WeatherSnow": [
       "will there be snow in Calgary/B-city",
        "does it snow in Oslo/B-city right now",
        "is there snow in Toronto/B-city today",
        "tell me if it is snowing in Ottawa/B-city",
        "are flurries falling in Winnipeg/B-city",
        "will i see snow in Edmonton/B-city",
        "check for snow in Reykjavik/B-city",
        "is it snowing in Denver/B-city",
        "will there be flurries in Prague/B-city",
        "check whether it is snowing in Moscow/B-city",
        "is snow falling in Helsinki/B-city",
        "tell me if there is snowfall in Zurich/B-city",
        "will it snow in Munich/B-city tonight",
        "is it snowy in Boston/B-city",
        "are there flurries in Stockholm/B-city",
        "check for snow in Minneapolis/B-city",
        "is it snowing in Sapporo/B-city",
        "tell me if it is snowing in Anchorage/B-city",
        "will there be snow in Fairbanks/B-city",
        "does Harbin/B-city have snow today",
        "is snow coming down in Krakow/B-city",
        "check for flurries in Geneva/B-city",
        "will Vienna/B-city get snow",
        "is it a snowy morning in Warsaw/B-city",
        "does Quebec/B-city have flurries right now",
        "tell me if Ulaanbaatar/B-city is getting snow",
        "will Reykjavik/B-city stay snowy today",
        "is there snowfall in Oslo/B-city tonight",
        "check if Montreal/B-city has flurries",
        "is there snowfall in Oslo/B-city tonight",
        "check if Montreal/B-city has flurries",
        "will it snow in Alaska/B-location right/B-time now/I-time",
        "is there snowfall in Ontario/B-location tomorrow/B-time",
        "check for flurries in Quebec/B-location this/B-time evening/I-time",
        "will Florida/B-location get snow this/B-time weekend/I-time",
        "is it going to snow in Colorado/B-location tonight/B-time",
        "tell me if there will be snow in Alberta/B-location next/B-time week/I-time"
    ],

    "Timer": [
    # Basic named timers
        "set a potato/B-name timer for 5/B-duration minutes/I-duration",
        "set timer for 10/B-duration minutes/I-duration called laundry/B-name",
        "start a pasta/B-name timer for 30/B-duration seconds/I-duration",
        "start timer for 2/B-duration hours/I-duration for the roast/B-name",
        "set a pizza/B-name timer for 45/B-duration minutes/I-duration",
        "egg/B-name timer for 15/B-duration minutes/I-duration",
        "set timer for 1/B-duration hour/I-duration labeled gym/B-name",
        "start a timer for 20/B-duration minutes/I-duration named study/B-name",
        "please set a tea/B-name timer for 3/B-duration hours/I-duration",
        "set timer for 90/B-duration seconds/I-duration for meditation/B-name",
        "set a 25/B-duration minute/I-duration cake/B-name timer",
        "oven/B-name timer for 40/B-duration minutes/I-duration",
        "start timer for 50/B-duration seconds/I-duration for the game/B-name",
        "set a timer for 6/B-duration hours/I-duration called sleep/B-name",
        "please start a workout/B-name timer for 12/B-duration minutes/I-duration",

        # Variations in phrasing and "fluff"
        "Can you put on a pizza/B-name timer for ten/B-duration minutes/I-duration please?",
        "I need a five/B-duration minute/I-duration nap/B-name countdown started now.",
        "Give me a 15/B-duration minute/I-duration shower/B-name timer.",
        "Hey, could you start a timer for 1/B-duration hour/I-duration and/I-duration 15/I-duration minutes/I-duration for the turkey/B-name?",
        "I'd like a chicken/B-name timer for 4/B-duration hours/I-duration.",

        # Written numbers vs. Digits
        "Timer for three/B-duration quarters/I-duration of/I-duration an/I-duration hour/I-duration for my break/B-name.",
        "Kick off a laundry/B-name timer for 12/B-duration minutes/I-duration.",
        "I need to time the plank/B-name for 60/B-duration seconds/I-duration.",
        "Go ahead and start a timer for 2/B-duration and/I-duration a/I-duration half/I-duration minutes/I-duration named boil/B-name.",
        "Let's do a 10/B-duration minute/I-duration focus/B-name timer.",

        # Different verbs and imperatives
        "Set a timer: 5/B-duration minutes/I-duration for the cookies/B-name.",
        "Time my run/B-name for 45/B-duration minutes/I-duration.",
        "Would you mind setting a timer for two/B-duration hours/I-duration called yardwork/B-name?",
        "Start counting down 15/B-duration seconds/I-duration for the start/B-name.",
        "Need a timer for 1/B-duration minute/I-duration for the microwave/B-name.",

        # Short/Abbreviated commands
        "Timer 20/B-duration mins/I-duration for beans/B-name.",
        "Set a 2/B-duration hour/I-duration grill/B-name timer.",
        "Start a timer for an/B-duration hour/I-duration called nap/B-name.",
        "Quick tea/B-name timer for 10/B-duration seconds/I-duration.",

        # Complex/Conversational
        "Create a timer for one/B-duration hour/I-duration and/I-duration twenty/I-duration minutes/I-duration for the dryer/B-name.",
        "I've got something in the oven, set a baking/B-name timer for 25/B-duration minutes/I-duration.",
        "Start a clock for 5/B-duration minutes/I-duration for the test/B-name.",
        "Run a timer for 120/B-duration seconds/I-duration for the drill/B-name.",
        "Remind me with a timer in 7/B-duration minutes/I-duration for the tea/B-name."
    ],
    
    "Alarm": [
        # Basic named and dated alarms
        "set a work/B-name alarm for 7/B-time am/I-time on Monday/B-day",
        "wake me at 6/B-time am/I-time this Tuesday/B-day for my flight/B-name",
        "set gym/B-name alarm for 8/B-time am/I-time next/B-day Friday/I-day",
        "wake me at 7/B-time 30/I-time am/I-time on Wednesday/B-day for school/B-name",
        "set an alarm at 9/B-time am/I-time tomorrow/B-day labeled doctor/B-name",
        "medicine/B-name alarm at 10/B-time pm/I-time every/B-day Sunday/I-day",
        "wake me at 6/B-time 45/I-time am/I-time on Thursday/B-day for the meeting/B-name",
        "set alarm for 5/B-time am/I-time called yoga/B-name for Saturday/B-day",
        "set an alarm for 11/B-time pm/I-time tonight/B-day named pill/B-name",
        "wake me at 8/B-time 15/I-time am/I-time on Monday/B-day for work/B-name",
        "alarm at 7/B-time am/I-time tomorrow/B-day for trash/B-name day",
        "set alarm at 6/B-time pm/I-time next/B-day Tuesday/I-day for the game/B-name",
        "wake me at 9/B-time am/I-time this/B-day Saturday/I-day for hiking/B-name",
        "set alarm for 10/B-time am/I-time called brunch/B-name for Sunday/B-day",
        "I need to wake up at 7/B-time am/I-time tomorrow/B-day for my commute/B-name",

        # Variations in phrasing and "fluff"
        "Can you set a laundry/B-name alarm for 6/B-time 30/I-time am/I-time next/B-day Monday/I-day?",
        "I need to be awake by 7/B-time am/I-time tomorrow/B-day for the interview/B-name.",
        "Could you put a workout/B-name alarm on for 8/B-time am/I-time this/B-day Friday/I-day?",
        "I've got a meeting/B-name, so set an alarm for 9/B-time 15/I-time am/I-time on Wednesday/B-day.",
        "Hey, wake me up at 6/B-time in/I-time the/I-time morning/I-time tomorrow/B-day for the trip/B-name.",

        # Written numbers and relative days
        "Set an alarm for seven/B-time thirty/I-time tomorrow/B-day morning/I-day for my run/B-name.",
        "Wake me at quarter/B-time past/I-time six/I-time am/I-time next/B-day Thursday/I-day for the sunrise/B-name.",
        "I need a shift/B-name alarm for eight/B-time o'clock/I-time on Saturday/B-day.",
        "Put a morning/B-name alarm for five/B-time am/I-time sharp this/B-day Sunday/I-day.",
        "Create a 10/B-time pm/I-time tonight/B-day alarm called medication/B-name.",

        # Short/Abbreviated commands
        "Work/B-name alarm 7/B-time am/I-time Monday/B-day.",
        "Set 6/B-time 15/I-time am/I-time tomorrow/B-day gym/B-name alarm.",
        "8/B-time am/I-time Sunday/B-day church/B-name alarm.",
        "New alarm: 10/B-time pm/I-time tonight/B-day for baking/B-name.",

        # Different verbs and imperatives
        "Alert me at 7/B-time 45/I-time am/I-time tomorrow/B-day for my appointment/B-name.",
        "Initialize a study/B-name alarm for 11/B-time pm/I-time this/B-day Friday/I-day.",
        "Get me up at 5/B-time 30/I-time am/I-time next/B-day Monday/I-day for work/B-name.",
        "Don't let me sleep past 6/B-time 30/I-time am/I-time tomorrow/B-day for the flight/B-name.",
        "I'd like to set a 7/B-time am/I-time alarm for Monday/B-day named office/B-name."
    ],



}

# Parsing Logic
def parse_example(sentence):
    tokens, slots = [], []
    for word in sentence.split():
        if "/" in word:
            token, slot = word.rsplit("/", 1)
            slot = slot.rstrip(".,?!:;")
        else:
            token, slot = word, "O"
        tokens.append(token)
        slots.append(slot)
    return tokens, slots

# Model Architecture (Must match core/m4_intent.py)
class JointIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.encoder.config.hidden_size
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]
        
        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
            loss = intent_loss + slot_loss
            
        return loss, intent_logits, slot_logits

# Dataset Class
class JointDataset(Dataset):
    def __init__(self, encodings, slot_labels, intent_labels):
        self.encodings = encodings
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "slot_labels": torch.tensor(self.slot_labels[idx]),
            "intent_label": torch.tensor(self.intent_labels[idx]),
        }

    def __len__(self):
        return len(self.intent_labels)

if __name__ == "__main__":
    print("Preparing dataset and tokenizing...")
    all_tokens, all_slots, all_intents = [], [], []
    
    for intent_name, sentences in intent_data.items():
        if intent_name in EXCLUDED_INTENTS:
            continue
        for sentence in sentences:
            tokens, slots = parse_example(sentence)
            all_tokens.append(tokens)
            all_slots.append(slots)
            all_intents.append(intent_name)

    # Create Mappings
    unique_slots = sorted(list(set(s for seq in all_slots for s in seq)))
    slot2id = {s: i for i, s in enumerate(unique_slots)}
    
    unique_intents = sorted(list(set(all_intents)))
    intent2id = {s: i for i, s in enumerate(unique_intents)}
    
    intent_labels_ids = [intent2id[i] for i in all_intents]

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(all_tokens, is_split_into_words=True, padding=True, truncation=True, max_length=16, return_tensors="pt")
    
    # Align labels for WordPiece Tokenizer
    aligned_slot_labels = []
    for i in range(len(all_tokens)):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(slot2id[all_slots[i][word_id]])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        aligned_slot_labels.append(label_ids)

    dataset = JointDataset(encodings, aligned_slot_labels, intent_labels_ids)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Building model and starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointIntentSlotModel(len(intent2id), len(slot2id)).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels = batch["slot_labels"].to(device)

            loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10 Loss: {total_loss:.3f}")

    # Save Model and Mappings
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "joint_bert.pt"))
    with open(os.path.join(MODEL_DIR, "intent2id.json"), 'w') as f:
        json.dump(intent2id, f)
    with open(os.path.join(MODEL_DIR, "slot2id.json"), 'w') as f:
        json.dump(slot2id, f)
        
    print(f"\nModel and mappings saved to {MODEL_DIR} successfully!")
