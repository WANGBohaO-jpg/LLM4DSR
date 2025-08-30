import ast
import pdb
import random
import re
import fire
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pdb

import json
from tqdm import tqdm
import os
import glob
import shutil

from accelerate import Accelerator
from accelerate.utils import gather_object, gather
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def batch(list, batch_size):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i : batch_size * (i + 1)]


def find_subsequence_indices(tensor, subsequence):
    sub_len = subsequence.size(0)

    windows = tensor.unfold(1, sub_len, 1)
    matches = (windows == subsequence).all(dim=2)

    indices = matches.max(dim=1).indices
    found = matches.any(dim=1)

    indices = torch.where(found, indices + sub_len - 1, torch.tensor(-1, device=tensor.device))

    return indices


def get_top_k_items(row, top_k):
    noise_prob = row["noise_prob"]
    item_ids = row["item_ids"]

    # 获取 noise_prob 的索引并按值从大到小排序
    sorted_indices = sorted(range(len(noise_prob)), key=lambda i: noise_prob[i], reverse=True)
    top_k_indices = sorted_indices[:top_k]

    # 根据索引获取对应的 item_ids 和 noise_prob
    top_k_items = [item_ids[i] for i in top_k_indices]
    top_k_probs = [noise_prob[i] for i in top_k_indices]

    return top_k_items, top_k_probs


def generate_prompt(instruction, input, output=None):
    if output:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
"""
    else:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def generate_suggest_prompt(item_ids, noise_item_id, id2name_dict):
    instruction = "You are to analyze a list of item titles provided by a user. Your task is to identify any item(s) that do not align with the main interests reflected by the majority of the items. After identifying these noise items, suggest alternative items that better match the user's interests."
    history = "User Interaction Sequence: " + ", ".join(f'"{id2name_dict[id]}"' for id in item_ids)
    history = f"{history}\n "
    output = f"Noise Items: '{id2name_dict[noise_item_id]}'\nSuggested Replacements: "

    return generate_prompt(instruction, history, output)


def generate_suggested_items(model, tokenizer, prompts, accelerator, num_beams=1, max_new_tokens=512):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)

    with torch.no_grad():
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generation_output = model.generate(**inputs, generation_config=generation_config)
        output_seq = generation_output.sequences
        output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)

        output = [_.split("\nSuggested Replacements: ")[-1].strip("\n").strip("'").strip('"').strip() for _ in output]
        output = [output[i * num_beams] for i in range(len(output) // num_beams)]

    return output


def main(
    denoise_model_path: str,
    denoise_dataset_path: str,
    batch_size: int = 8,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
    top_k: int = 3,
):
    set_seed(42)
    accelerator = Accelerator()

    id2name_path = os.path.join(denoise_dataset_path, "id2name4Rec.json")
    with open(id2name_path, "r") as file:
        id2name_dict = json.load(file)
    id2name_dict = {int(k): v for k, v in id2name_dict.items()}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    if denoise_model_path != "none":
        model = PeftModel.from_pretrained(model, denoise_model_path, torch_dtype=torch.bfloat16)
        model.merge_and_unload()
    model.eval()

    data_root_list = glob.glob(os.path.join(denoise_dataset_path, "*.csv"))
    data_root_list = [os.path.basename(file_path) for file_path in data_root_list]

    for file_name in data_root_list:
        test_flag = True if "test" in file_name else False

        data_path = os.path.join(denoise_dataset_path, file_name)
        df = pd.read_csv(data_path)

        if "suggest_item_titles" in df.columns or "noise_prob" not in df.columns:
            continue

        df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
        df["noise_prob"] = df["noise_prob"].apply(ast.literal_eval)

        item_ids_list = df["item_ids"].tolist()
        if test_flag:
            item_ids_list = [item_ids[:-1] for item_ids in item_ids_list]

        df["noise_items"], df["noise_items_prob"] = zip(*df.apply(get_top_k_items, axis=1, top_k=top_k))
        noise_items_list = df["noise_items"].tolist()

        prompt_list = []
        for item_ids, noise_item_ids in zip(item_ids_list, noise_items_list):
            for noise_item_id in noise_item_ids:
                prompt_list.append(generate_suggest_prompt(item_ids, noise_item_id, id2name_dict))

        with accelerator.split_between_processes(prompt_list) as prompts:
            suggested_items_list = []
            for batch_prompts in tqdm(batch(prompts, batch_size=batch_size), total=len(prompts) // batch_size):
                suggested_items = generate_suggested_items(model, tokenizer, batch_prompts, accelerator)
                suggested_items_list.extend(suggested_items)  # 长度为topk * data_num
        suggested_items_list = gather_object(suggested_items_list)

        if accelerator.is_main_process:
            result = []
            for i in range(0, len(suggested_items_list), top_k):
                sublist = suggested_items_list[i : i + top_k]
                result.append(sublist)

            df["suggest_item_titles"] = result
            df.to_csv(data_path, index=False)

        accelerator.wait_for_everyone()

    accelerator.print("All done!")


if __name__ == "__main__":
    fire.Fire(main)
