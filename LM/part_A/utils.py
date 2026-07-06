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
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)
    return input_ids, labels, n_tokens


def get_dataloaders(train_raw, dev_raw, test_raw, tokenizer, device, train_batch=8, eval_batch=16):
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device),
    )
    return train_loader, dev_loader, test_loader
