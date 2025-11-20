from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

################################################################################
#                             CARBON TRACKER                                   #
################################################################################
class CarbonTracker:
    """Track cumulative kg CO₂ based on power×carbon-intensity trace."""

    GPU_POWER_KW = 0.3  # 300 W nominal GPU power draw

    def __init__(self, trace_csv_path: str | None):
        if trace_csv_path and Path(trace_csv_path).exists():
            df = pd.read_csv(trace_csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.trace = df.set_index("timestamp")
        else:
            self.trace = None
        self.last_time = pd.Timestamp.utcnow()
        self.cumulative_kg = 0.0

    def _carbon_intensity(self):
        if self.trace is None:
            return 0.4  # default 0.4 kg/kWh global average
        now = pd.Timestamp.utcnow()
        if now in self.trace.index:
            return self.trace.loc[now]["gCO2_per_kWh"] / 1000
        idx = self.trace.index.get_indexer([now], method="nearest")[0]
        return self.trace.iloc[idx]["gCO2_per_kWh"] / 1000

    def update(self):
        now = pd.Timestamp.utcnow()
        hours = (now - self.last_time).total_seconds() / 3600.0
        self.last_time = now
        self.cumulative_kg += self.GPU_POWER_KW * hours * self._carbon_intensity()

################################################################################
#                               DATA LOADERS                                   #
################################################################################

def _tokenise_fn(examples: Dict, tokenizer, max_len: int):
    """Tokenise GSM8K Q-A pairs ensuring NO label tokens leak into inputs."""
    Q, A = examples["question"], examples["answer"]
    ids, attn, labels, rq, ra = [], [], [], [], []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    for q, a in zip(Q, A):
        prompt = q + "\nAnswer:"
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(a, add_special_tokens=False)["input_ids"] + [eos]

        # Truncate answer if needed so total sequence ≤ max_len
        a_avail = max_len - len(p_ids)
        a_ids = a_ids[: max(0, a_avail)]

        # Build input_ids WITHOUT answer tokens (data-leak prevention)
        seq = p_ids + [pad_id] * len(a_ids)
        if len(seq) < max_len:
            seq += [pad_id] * (max_len - len(seq))
        mask = [1] * len(p_ids) + [0] * (max_len - len(p_ids))

        # Build labels: predict first answer token after last prompt token
        lbl = [-100] * (len(p_ids) - 1) + a_ids
        lbl += [-100] * (max_len - len(lbl))

        ids.append(seq[:max_len])
        attn.append(mask[:max_len])
        labels.append(lbl[:max_len])
        rq.append(q)
        ra.append(a)
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "raw_question": rq,
        "raw_answer": ra,
    }


def build_dataloaders(cfg, tokenizer):
    cache = ".cache"
    config = getattr(cfg.dataset, 'config', None)
    ds_train = load_dataset(cfg.dataset.name, config, split=cfg.dataset.split, cache_dir=cache)
    ds_val = load_dataset(cfg.dataset.name, config, split=cfg.dataset.val_split, cache_dir=cache)

    cols_train, cols_val = ds_train.column_names, ds_val.column_names

    ds_train = ds_train.map(
        lambda x: _tokenise_fn(x, tokenizer, cfg.dataset.max_tokens),
        batched=True,
        remove_columns=cols_train,
    )
    ds_val = ds_val.map(
        lambda x: _tokenise_fn(x, tokenizer, cfg.dataset.max_tokens),
        batched=True,
        remove_columns=cols_val,
    )

    numeric = ["input_ids", "attention_mask", "labels"]
    all_cols = numeric + ["raw_question", "raw_answer"]
    ds_train.set_format(type="torch", columns=all_cols)
    ds_val.set_format(type="torch", columns=all_cols)

    def collate(batch: List[Dict]):
        out = {}
        for k in numeric:
            tensors = [b[k] if torch.is_tensor(b[k]) else torch.tensor(b[k]) for b in batch]
            pad_val = tokenizer.pad_token_id if k != "labels" else -100
            out[k] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=pad_val
            )
        out["raw_question"] = [b["raw_question"] for b in batch]
        out["raw_answer"] = [b["raw_answer"] for b in batch]
        return out

    train_dl = DataLoader(
        ds_train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_dl = DataLoader(
        ds_val,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return {"train": train_dl, "val": val_dl}