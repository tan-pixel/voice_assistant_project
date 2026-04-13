import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class JointIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.encoder.config.hidden_size
        
        # MLP Head for Intent
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
        # Standard Linear Head for Slots
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]
        
        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits

# The main Inference Module
class IntentDetectionModule:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Define paths where offline-trained models will be saved
        self.model_path = "data/models/joint_bert.pt"
        self.intent_map_path = "data/models/intent2id.json"
        self.slot_map_path = "data/models/slot2id.json"
        
        self.model_loaded = self._load_model()

    def _load_model(self):
        """Attempts to load the trained PyTorch model and mappings."""
        if not (os.path.exists(self.model_path) and os.path.exists(self.intent_map_path) and os.path.exists(self.slot_map_path)):
            print("WARNING: Trained BERT model files not found. Using fallback heuristics.")
            return False

        with open(self.intent_map_path, 'r') as f:
            self.intent2id = json.load(f)
            self.id2intent = {v: k for k, v in self.intent2id.items()}
            
        with open(self.slot_map_path, 'r') as f:
            self.slot2id = json.load(f)
            self.id2slot = {v: k for k, v in self.slot2id.items()}

        self.model = JointIntentSlotModel(len(self.intent2id), len(self.slot2id))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Successfully loaded Joint Intent & Slot BERT model.")
        return True

    def process(self, transcribed_text):
        """
        Input: Sentence from ASR module
        Output: Recognized Intent and required Slots
        """
        text = transcribed_text.lower().strip()
        
        if self.model_loaded:
            return self._predict_with_bert(text)
        else:
            raise FileNotFoundError("Trained BERT model not found. Please run the training script to generate the model files.")
        
    def _predict_with_bert(self, text):
        """Runs the actual DistilBERT inference on the input text."""
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=32).to(self.device)
        
        with torch.no_grad():
            intent_logits, slot_logits = self.model(encoding["input_ids"], encoding["attention_mask"])
            
            # Predict Intent
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            intent_name = self.id2intent[intent_idx]
            
            # Predict Slots
            slot_preds_idx = torch.argmax(slot_logits, dim=2)[0]
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            
            slots = {}
            current_slot_name = None
            current_slot_value = []
            
            # Extract BIO tags into a dictionary
            for idx, p_idx in enumerate(slot_preds_idx):
                # Skip special tokens ([CLS], [SEP], padding)
                if tokens[idx] in ["[CLS]", "[SEP]", "[PAD]"]: 
                    continue
                    
                slot_label = self.id2slot[p_idx.item()]
                token = tokens[idx]
                
                # Reconstruct words split by the WordPiece tokenizer (e.g., "##ing")
                if token.startswith("##"):
                    token = token[2:]
                    
                if slot_label.startswith("B-"):
                    if current_slot_name:
                        slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()
                    current_slot_name = slot_label[2:].lower()
                    current_slot_value = [token]
                elif slot_label.startswith("I-") and current_slot_name == slot_label[2:].lower():
                    current_slot_value.append(f" {token}")
                else:
                    if current_slot_name:
                        slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()
                        current_slot_name = None
                        current_slot_value = []
            
            # Catch the last slot if the sentence ended on an entity
            if current_slot_name:
                slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()

            return {"intent": intent_name, "slots": slots}