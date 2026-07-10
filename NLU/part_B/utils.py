import json
from collections import Counter
from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0
IGNORE_ID = -100  # CrossEntropyLoss default ignore_index: special/pad/continuation subtokens


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def create_dev_set(tmp_train_raw, portion=0.10):
    """10% of train as dev, stratified on intent (same split as 2.A).

    Intents occurring once cannot be stratified: they go straight into train.
    """
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    inputs = []
    labels = []
    mini_train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, _, _ = train_test_split(
        inputs, labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    X_train.extend(mini_train)
    return X_train, X_dev


class Lang():
    """Label mappings only: word2id is replaced by the HF tokenizer.

    slot2id keeps 'pad'=0 as in 2.A, but ignored positions use IGNORE_ID here.
    """

    def __init__(self, intents, slots):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    """Word-level samples; subword tokenization is deferred to collate_fn."""

    def __init__(self, dataset, lang):
        self.utterances = []
        self.slots = []
        self.intents = []

        for x in dataset:
            self.utterances.append(x['utterance'].split())
            self.slots.append(x['slots'].split())
            self.intents.append(x['intent'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return {
            'utterance': self.utterances[idx],  # list of words, not joined
            'slots': self.slot_ids[idx],        # one id per word
            'intent': self.intent_ids[idx],
        }

    def mapping_lab(self, data, mapper):
        return [mapper[x] for x in data]

    def mapping_seq(self, data, mapper):
        return [[mapper[x] for x in seq] for seq in data]


def collate_fn(data, tokenizer, device):
    utterances = [d['utterance'] for d in data]
    # tokenize pre-split words: word_ids() gives the exact subtoken→word map
    enc = tokenizer(utterances, is_split_into_words=True, padding=True, return_tensors='pt')

    # each word's label goes on its FIRST subtoken; special tokens, padding
    # and continuation subtokens get IGNORE_ID (arXiv:1902.10909)
    y_slots = torch.full(enc.input_ids.shape, IGNORE_ID, dtype=torch.long)
    for i, d in enumerate(data):
        prev = None
        for j, wid in enumerate(enc.word_ids(batch_index=i)):
            if wid is not None and wid != prev:
                y_slots[i, j] = d['slots'][wid]
            prev = wid

    intent = torch.LongTensor([d['intent'] for d in data])
    # real (non-pad) subtokens per sequence; the GPT2 intent head reads slots_len-1
    slots_len = enc.attention_mask.sum(dim=1)

    new_item = {}
    new_item['utterances'] = enc.input_ids.to(device)
    new_item['attention_mask'] = enc.attention_mask.to(device)
    new_item['y_slots'] = y_slots.to(device)
    new_item['intents'] = intent.to(device)
    new_item['slots_len'] = slots_len.to(device)
    new_item['words'] = utterances  # original words, needed by conll in eval
    return new_item


def get_dataloaders(train_raw, dev_raw, test_raw, lang, tokenizer, device, train_batch=32, eval_batch=64):
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

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
