import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# The Neural Architecture 
class JointIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased", local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]
        
        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits

# The Main Inference Module
class IntentDetectionModule:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
        
        self.model_path = "data/models/joint_bert.pt"
        self.intent_map_path = "data/models/intent2id.json"
        self.slot_map_path = "data/models/slot2id.json"
        
        self.model_loaded = self._load_model()

    def _load_model(self):
        """Loads the trained PyTorch model and dynamic mappings."""
        if not (os.path.exists(self.model_path) and os.path.exists(self.intent_map_path) and os.path.exists(self.slot_map_path)):
            print("WARNING: Trained BERT model files not found.")
            return False
        
        with open(self.intent_map_path, 'r') as f:
            self.intent2id = json.load(f)
            
        with open(self.slot_map_path, 'r') as f:
            self.slot2id = json.load(f)
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        model_intents = state_dict["intent_classifier.3.weight"].shape[0]
        model_slots = state_dict["slot_classifier.weight"].shape[0]

        self.id2intent = {
            idx: name for name, idx in self.intent2id.items() if idx < model_intents
        }
        self.id2slot = {
            idx: name for name, idx in self.slot2id.items() if idx < model_slots
        }

        if len(self.id2intent) != model_intents or len(self.id2slot) != model_slots:
            raise ValueError("Saved mappings do not cover the loaded model dimensions.")

        self.model = JointIntentSlotModel(model_intents, model_slots)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Successfully loaded Joint Intent & Slot BERT model.")
        return True

    def process(self, transcribed_text):
        """
        Input: Sentence from ASR module
        Output: Recognized Intent and required Slots directly from BERT
        """
        text = transcribed_text.lower().strip()
        
        if not self.model_loaded:
            raise FileNotFoundError("Trained BERT model not found. Run the training script.")

        result = self._predict_with_bert(text)
        result["raw_text"] = text 
        return result

    def _predict_with_bert(self, text):
        """Runs DistilBERT inference and decodes BIO tags into a dictionary."""
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
            
            # Extract BIO tags
            for idx, p_idx in enumerate(slot_preds_idx):
                if tokens[idx] in ["[CLS]", "[SEP]", "[PAD]"]: 
                    continue
                    
                slot_label = self.id2slot[p_idx.item()]
                token = tokens[idx]
                
                # Reconstruct WordPieces (e.g., "##ing")
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
            
            if current_slot_name:
                slots[current_slot_name] = "".join(current_slot_value).replace(" ", " ").strip()

            return {"intent": intent_name, "slots": slots}