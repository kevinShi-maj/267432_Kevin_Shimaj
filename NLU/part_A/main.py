from functions import *
from model import GPT2
from utils import PAD_TOKEN, load_data, create_dev_set, Lang, get_dataloaders

import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- experiment config -------------------------------------------------
    # Step 0 closed: lr=5e-3. Step 1.1 closed: d_model=128 (256 loses even with
    # adjusted lr). Step 1.2: n_heads ladder at head_dim = d_model / n_heads.
    lr = 5e-3
    d_model = 128
    n_heads = 4
    num_layers = 1
    ff_dim = 20
    dropout = 0.0  # Step 2: 0.1, then 0.2

    # single seed for the incremental ladder; final protocol: [0, 1, 2, 3, 4]
    seeds = [42]

    n_epochs = 200
    # lab default is 3, but one ATIS epoch = ~35 updates: rescaled to 10 based on
    # the longest no-improvement stall (10 epochs) seen in the diagnostic F1 curve
    patience_max = 10
    # ------------------------------------------------------------------------

    tmp_train_raw = load_data(os.path.join("dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))
    train_raw, dev_raw = create_dev_set(tmp_train_raw)

    # words from train only (test unknowns become 'unk');
    # slot/intent labels from the whole corpus (no unk wanted on labels)
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, lang, DEVICE
    )

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)  # pad and cls share the same id
    n_intents = len(lang.intent2id)

    slot_f1s, intent_accs = [], []
    best_dev_f1_overall = 0
    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = GPT2(
            vocab_len,
            slots_len,
            n_intents,
            pos_emb_size=1024,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        ).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # skips pad AND cls positions
        criterion_intents = nn.CrossEntropyLoss()

        patience = patience_max
        best_f1 = 0
        best_model = None
        epochs_run = 0
        f1_history = []
        pbar = tqdm(range(n_epochs))

        for epoch in pbar:
            epochs_run = epoch + 1
            train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            results_dev, intent_dev, _ = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            f1_dev = results_dev['total']['f']
            f1_history.append(round(f1_dev, 4))
            pbar.set_description(
                "Seed %d | Dev Slot F1: %.4f | Dev Intent Acc: %.4f"
                % (seed, f1_dev, intent_dev['accuracy'])
            )

            # early stopping on dev slot F1 (higher = better)
            if f1_dev > best_f1:
                best_f1 = f1_dev
                best_model = copy.deepcopy(model).to("cpu")
                patience = patience_max
            elif f1_dev > 0:
                # F1 = 0 = collapsed all-O phase, the model cannot be judged yet:
                # those epochs do not consume patience
                patience -= 1
            if patience <= 0:
                break

        print("Seed %d — dev Slot F1 per epoch: %s" % (seed, f1_history))

        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang
        )
        slot_f1s.append(results_test['total']['f'])
        intent_accs.append(intent_test['accuracy'])
        print(
            "Seed %d — epochs: %d | best dev Slot F1: %.4f | test Slot F1: %.4f | test Intent Acc: %.4f"
            % (seed, epochs_run, best_f1, slot_f1s[-1], intent_accs[-1])
        )

        # keep the weights of the run with the best dev F1
        if best_f1 > best_dev_f1_overall:
            best_dev_f1_overall = best_f1
            os.makedirs("bin", exist_ok=True)
            torch.save(best_model.state_dict(), "bin/best_model.pt")

    slot_f1s = np.asarray(slot_f1s)
    intent_accs = np.asarray(intent_accs)
    print("Test Slot F1:     %.4f +- %.4f" % (slot_f1s.mean(), slot_f1s.std()))
    print("Test Intent Acc:  %.4f +- %.4f" % (intent_accs.mean(), intent_accs.std()))
