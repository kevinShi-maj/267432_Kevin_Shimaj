# Part 1.B — fine-tune pre-trained GPT2 on Penn TreeBank with hand-written LoRA.
# Run:  python main.py   (from inside LM/part_B/)
# Target: test PPL < 250 AND test PPL(1.B) < test PPL(1.A) = 33.68.

from functions import param_stats, freeze_non_lora, train_loop, eval_loop
from model import GPT2_LoRA
from utils import read_file, get_dataloaders

import math
import copy
import os
import random

import numpy as np
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from tqdm import tqdm


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- LoRA hyperparameters (incremental grid in MEMORY.md §1.B) ---
    RANK = 8            # Step 1 (rank): grid 4, 8, 16, 32 — sweep AFTER lr is fixed
    ALPHA = 8           # Step 2 (alpha): grid rank, 2*rank — kept = rank during lr sweep
    lr = 1e-3           # Step 0 (lr sweep) @ r=8/a=8: 5e-4 -> dev 23.36/test 21.11 (B1). Now 1e-3, then 1e-4.
    train_batch, eval_batch = 16, 16   # drop to 8 if CUDA OOM

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw   = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw  = read_file("dataset/PennTreeBank/ptb.test.txt")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token (mandatory)

    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, tokenizer, DEVICE, train_batch, eval_batch
    )

    # Build pre-trained GPT2, swap in LoRA attention blocks.
    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=RANK, alpha=ALPHA).to(DEVICE)

    # Train ONLY the LoRA adapters.
    freeze_non_lora(model)
    print(f"\n=== LoRA config: rank={RANK}, alpha={ALPHA}, scaling={ALPHA / RANK:g}, lr={lr} ===")
    param_stats(model)
    expected = model.config.n_layer * 6 * RANK * model.config.n_embd
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == expected, f"trainable {trainable} != expected {expected} (n_layer*6*rank*d_model)"

    # Sanity: ΔW must be 0 at init (B=0), so 1.B starts exactly at the pre-trained model.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            assert torch.count_nonzero(p).item() == 0, f"{name} is not zero-initialised — ΔW != 0 at start!"
    print("check OK: all lora_B zero-initialised (ΔW = 0 at step 0)\n")

    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    n_epochs = 20
    patience = 3
    best_ppl = math.inf
    best_model = None
    best_epoch = 0
    pbar = tqdm(range(n_epochs))

    for epoch in pbar:
        train_loop(train_loader, optimizer, model)
        ppl_dev, _ = eval_loop(dev_loader, model)
        pbar.set_description("Dev PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to("cpu")
            best_epoch = epoch + 1
            patience = 3
        else:
            patience -= 1
        if patience <= 0:
            break

    os.makedirs("bin", exist_ok=True)
    torch.save(best_model.state_dict(), "bin/best_model.pt")
    print(f"Epochs run: {epoch + 1} (best at epoch {best_epoch})")
    print("Best dev PPL:", best_ppl)

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, best_model)
    print("Test PPL:", final_ppl)
    print("Target check: PPL < 250 ->", final_ppl < 250, "| PPL < 1.A (33.68) ->", final_ppl < 33.68)
