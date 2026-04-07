import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Configurations
MODEL_DIR = "../data/models"

# Define the Dataset for D&D, Game, and Basic Intents
# Using the BIO inline annotation format
intent_data = {
    "lookup_monster": [
        "tell me about goblins/B-monster_name",
        "what is a dragon/B-monster_name",
        "look up stats for a zombie/B-monster_name"
    ],
    "lookup_spell": [
        "what does fireball/B-spell_name do",
        "how much damage does magic/B-spell_name missile/I-spell_name deal",
        "explain the shield/B-spell_name spell"
    ],
    "equipment_category": [
        "show me martial/B-category weapons/I-category",
        "list all simple/B-category weapons/I-category",
        "what is in the armor/B-category category"
    ],
    "move": [
        "move north/B-direction",
        "go south/B-direction",
        "walk east/B-direction",
        "head west/B-direction"
    ],
    "pick_up": [
        "pick up the sword/B-item_name",
        "grab the key/B-item_name",
        "take the health/B-item_name potion/I-item_name"
    ],
    "use_item": [
        "use a potion/B-item_name",
        "drink the health/B-item_name potion/I-item_name"
    ],
    "restart_game": [
        "restart the game",
        "start over",
        "reset the dungeon"
    ],
    "Greetings": ["hello", "hi there", "hey atlas"],
    "Goodbye": ["thank you", "goodbye", "bye"],
    "OOS": ["do my grocery shopping", "play some music", "turn off the lights"],
    "Weather": [
        "what is the weather in Paris/B-city",
        "does it rain in London/B-city today",
        "temperature in Tokyo/B-city"
    ],
    "Timer": [
        "set a soup/B-name timer for 2/B-duration minutes/I-duration",
        "start a 5/B-duration minute/I-duration timer",
        "set alarm for 10/B-duration seconds/I-duration"
    ],
}

# Parsing Logic
def parse_example(sentence):
    tokens, slots = [], []
    for word in sentence.split():
        if "/" in word:
            token, slot = word.rsplit("/", 1)
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