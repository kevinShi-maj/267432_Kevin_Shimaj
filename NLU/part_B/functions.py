import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from conll import evaluate
from utils import IGNORE_ID


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []

    for batch in data:
        optimizer.zero_grad()
        slots, intent = model(batch['utterances'], batch['attention_mask'], batch['slots_len'])
        slots = slots.permute(0, 2, 1)  # CrossEntropyLoss wants (B, C, L)

        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for batch in data:
            slots, intents = model(batch['utterances'], batch['attention_mask'], batch['slots_len'])
            slots = slots.permute(0, 2, 1)

            loss_intent = criterion_intents(intents, batch['intents'])
            loss_slot = criterion_slots(slots, batch['y_slots'])
            loss_array.append((loss_intent + loss_slot).item())

            # intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # slot inference: predictions read ONLY at first-subtoken positions
            # (y_slots != IGNORE_ID), mapped back to words so refs and hyps
            # carry identical word sequences for conll
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                valid = batch['y_slots'][id_seq] != IGNORE_ID
                gt_ids = batch['y_slots'][id_seq][valid].tolist()
                to_decode = seq[valid].tolist()
                utterance = batch['words'][id_seq]  # len(utterance) == valid.sum()

                ref_slots.append([(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(gt_ids)])
                hyp_slots.append([(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # early in training the model can predict a label absent from the
        # references, which conll cannot score; self-resolves as training goes on
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array
