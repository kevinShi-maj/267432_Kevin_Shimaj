# Part 2.A — ATIS loading and preprocessing (lab-05 pipeline).

import json
from collections import Counter
from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def create_dev_set(tmp_train_raw, portion=0.10):
    """10% of train as dev, stratified on intent so the label distribution is kept.

    Intents occurring once cannot be stratified, so they go straight into train.
    random_state is fixed: every experiment of the ladder sees the same split.
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
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        # cls shares the pad id, so it is dropped here not to shadow 'pad'
        self.id2slot = {v: k for k, v in self.slot2id.items() if not cls or k != 'cls'}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        if cls:
            vocab['cls'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True, cls=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            # cls is no real word: giving its slot label the pad id makes the
            # slot loss (ignore_index=PAD_TOKEN) skip that position
            vocab['cls'] = PAD_TOKEN
        return vocab


class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        return {'utterance': utt, 'slots': slots, 'intent': intent}

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            if self.add_cls:
                # cls appended at the END: with the causal mask the last position
                # is the only one that has attended to the whole sentence
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res


def collate_fn(data, device):
    def merge(sequences):
        # right-pad every sequence with PAD_TOKEN to the batch max length
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths

    data_by_key = {}
    for key in data[0].keys():
        data_by_key[key] = [d[key] for d in data]

    # one length per sequence is enough: len(utterance) == len(slots)
    src_utt, _ = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key['slots'])
    intent = torch.LongTensor(data_by_key['intent'])

    return {
        'utterances': src_utt.to(device),
        'intents': intent.to(device),
        'y_slots': y_slots.to(device),
        'slots_len': torch.LongTensor(y_lengths).to(device),
    }


def get_dataloaders(train_raw, dev_raw, test_raw, lang, device, train_batch=128, eval_batch=64):
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        collate_fn=partial(collate_fn, device=device),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch,
        collate_fn=partial(collate_fn, device=device),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch,
        collate_fn=partial(collate_fn, device=device),
    )
    return train_loader, dev_loader, test_loader
