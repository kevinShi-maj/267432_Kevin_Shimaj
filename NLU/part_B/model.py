import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model


class BertForNLU(nn.Module):
    def __init__(self, n_slots, n_intents, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        d_model = self.bert.config.hidden_size
        self.slot_out = nn.Linear(d_model, n_slots)
        self.intent_out = nn.Linear(d_model, n_intents)

    # slots_len accepted (and ignored) so both models share the same call signature
    def forward(self, input_ids, attention_mask, slots_len=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state              # (B, L, d_model)
        slots = self.slot_out(seq)               # (B, L, n_slots)
        intent = self.intent_out(seq[:, 0])      # [CLS] is at position 0
        return slots, intent


class GPT2ForNLU(nn.Module):
    def __init__(self, n_slots, n_intents, model_name="openai-community/gpt2"):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        d_model = self.gpt2.config.n_embd
        self.slot_out = nn.Linear(d_model, n_slots)
        self.intent_out = nn.Linear(d_model, n_intents)

    def forward(self, input_ids, attention_mask, slots_len):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        slots = self.slot_out(seq)
        # intent from the last real (non-pad) token; position 0 has seen nothing
        # under the causal mask, and pad positions carry no information
        cls_tokens = torch.stack([seq[i, slots_len[i] - 1] for i in range(seq.shape[0])])
        intent = self.intent_out(cls_tokens)
        return slots, intent
