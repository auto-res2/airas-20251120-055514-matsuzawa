from typing import Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = ".cache"


def load_model_and_tokenizer(model_cfg) -> Tuple[AutoTokenizer, torch.nn.Module]:
    name = model_cfg.name
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR, padding_side="right")
    # Ensure pad token exists ---------------------------------------------------
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    dtype = (
        torch.float16
        if getattr(model_cfg, "precision", "fp32").lower() in ["fp16", "float16"]
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR, torch_dtype=dtype)

    # Gradient checkpointing ----------------------------------------------------
    if getattr(model_cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # LoRA ---------------------------------------------------------------------
    if hasattr(model_cfg, "lora"):
        l_cfg = model_cfg.lora
        conf = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=l_cfg.r,
            lora_alpha=l_cfg.alpha,
            lora_dropout=l_cfg.dropout,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, conf)

    model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model