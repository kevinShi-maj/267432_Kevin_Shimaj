# Data loading and preprocessing for Part 1.B: same Penn TreeBank pipeline as 1.A.

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    # GPT2LMHeadModel applies the causal shift itself (logits[:, :-1] vs labels[:, 1:]),
    # so labels are an unshifted copy of the input, unlike in 1.A where we shift by hand.
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Number of positions the loss is actually averaged over, needed to weight each
    # batch when aggregating perplexity across the epoch (same convention as 1.A).
    n_tokens = (labels[:, 1:] != -100).sum()
    return input_ids, attention_mask, labels, n_tokens


def get_dataloaders(train_raw, dev_raw, test_raw, tokenizer, device, train_batch=16, eval_batch=16):
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    cf = partial(collate_fn, tokenizer=tokenizer, device=device)
    train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=cf, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch, collate_fn=cf)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch, collate_fn=cf)
    return train_loader, dev_loader, test_loader
