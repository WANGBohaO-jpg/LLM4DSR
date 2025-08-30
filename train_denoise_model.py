import re
import os
import random
from typing import List, Optional

import fire
import numpy as np
import torch
import transformers
from datasets import load_dataset
from accelerate import Accelerator

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(
    dataset_name: str,
    manual_noise: float = -1,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
):
    set_seed(seed)
    if manual_noise != -1:
        train_data_path = os.path.join(
            "./data", dataset_name, f"TDMD_noise_{manual_noise}", "train.json"
        )
        output_dir = os.path.join(
            "./save_denoise_model",
            dataset_name,
            f"batch{batch_size}_sample{sample}_epoch{num_epochs}_manualnoise{manual_noise}",
        )
    else:
        train_data_path = os.path.join("./data", dataset_name, f"TDMD", "train.json")
        output_dir = os.path.join(
            "./save_denoise_model",
            dataset_name,
            f"batch{batch_size}_sample{sample}_epoch{num_epochs}",
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size = batch_size // world_size
    gradient_accumulation_steps = batch_size // micro_batch_size

    if world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    train_data = load_dataset("json", data_files=train_data_path, split="train")
    train_data = train_data.shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data.shuffle(seed=seed)
    train_data = train_data.map(lambda x: generate_and_tokenize_prompt(x))

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=val_data,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=0.1,
            # evaluation_strategy="steps",
            # eval_steps=0.1,
            save_strategy="steps",
            save_steps=(1 / num_epochs),
            # save_total_limit=10,
            save_on_each_node=False,
            log_on_each_node=False,
            # load_best_model_at_end=True,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir, save_embedding_layers=False)


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
