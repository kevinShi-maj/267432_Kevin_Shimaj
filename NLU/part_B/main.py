from functions import *
from model import BertForNLU, GPT2ForNLU
from utils import IGNORE_ID, load_data, create_dev_set, Lang, get_dataloaders

import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from tqdm import tqdm


MODEL_CONFIGS = {
    "bert": {
        "model_name": "bert-base-uncased",
        "cls": BertForNLU,
        "lr": 2e-5,               # grid: {1e-5, 2e-5, 3e-5, 5e-5}
        "tokenizer_kwargs": {},
    },
    "gpt2": {
        "model_name": "openai-community/gpt2",
        "cls": GPT2ForNLU,
        # fast tokenizer needs add_prefix_space=True for pre-split words
        "lr": 1e-4,               # grid: {5e-5, 1e-4, 2e-4}
        "tokenizer_kwargs": {"add_prefix_space": True},
    },
}


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- experiment config -------------------------------------------------
    models_to_run = ["bert", "gpt2"]

    # single seed for the lr grid; final protocol: [0, 1, 2, 3, 4]
    seeds = [42]

    train_batch, eval_batch = 32, 64
    n_epochs = 30
    # 140 updates/epoch and a pre-trained start: convergence expected < 10 epochs,
    # no collapse phase — lab patience works here (F1=0 guard kept just in case)
    patience_max = 3
    # ------------------------------------------------------------------------

    tmp_train_raw = load_data(os.path.join("dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))
    train_raw, dev_raw = create_dev_set(tmp_train_raw)

    # labels from the whole corpus (no unk wanted on labels), as in 2.A
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(intents, slots)

    slots_len = len(lang.slot2id)
    n_intents = len(lang.intent2id)

    for model_key in models_to_run:
        cfg = MODEL_CONFIGS[model_key]
        print("\n===== %s (%s) — lr %g =====" % (model_key, cfg['model_name'], cfg['lr']))

        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], **cfg['tokenizer_kwargs'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # GPT2 ships no pad token

        train_loader, dev_loader, test_loader = get_dataloaders(
            train_raw, dev_raw, test_raw, lang, tokenizer, DEVICE,
            train_batch=train_batch, eval_batch=eval_batch,
        )

        slot_f1s, intent_accs = [], []
        best_dev_f1_overall = 0
        for seed in seeds:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # full fine-tune: all params trainable, fresh heads on top;
            # no init_weights here — it would wipe the pre-trained weights
            model = cfg['cls'](slots_len, n_intents, model_name=cfg['model_name']).to(DEVICE)

            optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'])
            criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
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
                    "%s seed %d | Dev Slot F1: %.4f | Dev Intent Acc: %.4f"
                    % (model_key, seed, f1_dev, intent_dev['accuracy'])
                )

                # early stopping on dev slot F1 (higher = better)
                if f1_dev > best_f1:
                    best_f1 = f1_dev
                    best_model = copy.deepcopy(model).to("cpu")
                    patience = patience_max
                elif f1_dev > 0:
                    patience -= 1
                if patience <= 0:
                    break

            print("%s seed %d — dev Slot F1 per epoch: %s" % (model_key, seed, f1_history))

            best_model.to(DEVICE)
            results_test, intent_test, _ = eval_loop(
                test_loader, criterion_slots, criterion_intents, best_model, lang
            )
            slot_f1s.append(results_test['total']['f'])
            intent_accs.append(intent_test['accuracy'])
            print(
                "%s seed %d — epochs: %d | best dev Slot F1: %.4f | test Slot F1: %.4f | test Intent Acc: %.4f"
                % (model_key, seed, epochs_run, best_f1, slot_f1s[-1], intent_accs[-1])
            )

            # keep the weights of the run with the best dev F1
            if best_f1 > best_dev_f1_overall:
                best_dev_f1_overall = best_f1
                os.makedirs("bin", exist_ok=True)
                torch.save(best_model.state_dict(), "bin/best_model_%s.pt" % model_key)

            # free GPU memory before the next run (two full models in one script)
            del model, best_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        slot_f1s = np.asarray(slot_f1s)
        intent_accs = np.asarray(intent_accs)
        print("%s — Test Slot F1:     %.4f +- %.4f" % (model_key, slot_f1s.mean(), slot_f1s.std()))
        print("%s — Test Intent Acc:  %.4f +- %.4f" % (model_key, intent_accs.mean(), intent_accs.std()))
