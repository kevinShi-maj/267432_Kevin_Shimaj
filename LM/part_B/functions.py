# Training / evaluation loops and helpers for Part 1.B.
# The model is a HuggingFace GPT2LMHeadModel subclass, so it returns the loss
# itself (output.loss) — we do NOT build a separate criterion here, unlike 1.A.
# PPL is aggregated token-weighted across batches (same convention as 1.A) so the
# two parts' perplexities are directly comparable.

import math
import torch
from tqdm import tqdm


def param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}  ({100 * trainable / total:.3f}%)")
    print(f"frozen params:    {total - trainable:,}")
    return total, trainable


def freeze_non_lora(model):
    """Freeze everything, then unfreeze only the LoRA adapters.

    LoRA fine-tuning trains ONLY the injected low-rank matrices; the whole
    pre-trained GPT2 stays frozen (Hu et al. §4.1). Every trainable parameter
    name therefore contains "lora_".
    """
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True


def train_loop(data, optimizer, model):
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))
    for i, (input_ids, attention_mask, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array) / sum(number_of_tokens)).item())

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, model):
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, n_tokens in tqdm(data, desc="Evaluating:", unit="batch", total=len(data)):
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return
