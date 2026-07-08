from functions import *
from model import GPT2
from utils import read_file, get_dataloaders

import math
import copy
import os

import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw   = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw  = read_file("dataset/PennTreeBank/ptb.test.txt")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, tokenizer, DEVICE
    )

    vocab_len = len(tokenizer)
    lr = 1e-3 # best LR found 
    model = GPT2(
        vocab_len,
        pos_emb_size=1024,
        d_model=256,  # Step 1.1 chiuso: 20 -> 128 (45.68) -> 256 (45.53, KEEP) -> 384 (45.73, revert)
        n_heads=8,    # Step 1.2: ladder 1 -> 4 (43.55, keep) -> 8 (head_dim 256/8 = 32)
        num_layers=1,
        ff_dim=20,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval  = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    n_epochs = 100
    patience = 3
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(n_epochs))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
        pbar.set_description("PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to("cpu")
            patience = 3
        else:
            patience -= 1

        if patience <= 0:
            break

    os.makedirs("bin", exist_ok=True)
    torch.save(best_model.state_dict(), "bin/best_model.pt")
    print("Best dev PPL:", best_ppl)

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print("Test PPL:", final_ppl)
