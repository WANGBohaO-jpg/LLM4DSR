import ast
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
import shutil

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def extract_info(path):
    match = re.search(r"_sample(\d+)_", path)
    if match:
        sample_number = match.group(1)
    else:
        sample_number = ""

    match = re.search(r"checkpoint-(\d+)", path)
    if match:
        checkpoint_number = match.group(1)
    else:
        checkpoint_number = None

    return sample_number, checkpoint_number


def find_subsequence_indices(tensor, subsequence):
    sub_len = subsequence.size(0)

    windows = tensor.unfold(1, sub_len, 1)
    matches = (windows == subsequence).all(dim=2)

    indices = matches.max(dim=1).indices
    found = matches.any(dim=1)

    indices = torch.where(found, indices + sub_len - 1, torch.tensor(-1, device=tensor.device))

    return indices


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


def generate_noise_prompt(item_ids, id2name_dict):
    instruction = "You are to analyze a list of item titles provided by a user. Your task is to identify any item(s) that do not align with the main interests reflected by the majority of the items. After identifying these noise items, suggest alternative items that better match the user's interests."
    history = "User Interaction Sequence: " + ", ".join(f'"{id2name_dict[id]}"' for id in item_ids)
    history = f"{history}\n "

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{history}

### Response:
Noise Items: '"""


def generate_first_tokenid(tokenizer, id2name_dict):
    tokenizer.padding_side = "right"
    item_titles = list(id2name_dict.values())
    token_ids = tokenizer(item_titles, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    first_token_ids_list = token_ids[:, 1].tolist()
    tokenizer.padding_side = "left"

    return first_token_ids_list


def get_noise_items_prob(first_token_ids, logits):
    logits = logits[:, -1, :]  # 8 x 128257
    log_probs = F.log_softmax(logits, dim=-1)  # 8 x 128257

    noise_prob = []
    for i, indices in enumerate(first_token_ids):
        indices_tensor = torch.tensor(indices)
        row_values = log_probs[i, indices_tensor]
        noise_prob.append(row_values.tolist())

    return noise_prob


def main(
    denoise_model_path: str,
    noise_dataset_path: str,
    denoise_data: str,
    sample: int,
    batch_size: int = 16,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
):
    set_seed(42)
    accelerator = Accelerator()

    if "noise" in noise_dataset_path:
        save_denoise_data_path = noise_dataset_path.replace("noise", "denoise")

    sample_number, checkpoint_num = extract_info(denoise_model_path)
    if checkpoint_num is not None:
        save_denoise_data_path = save_denoise_data_path + "_" + sample_number + "_" + checkpoint_num
    else:
        save_denoise_data_path = save_denoise_data_path + "_" + sample_number

    if accelerator.is_main_process:
        if not os.path.exists(save_denoise_data_path):
            os.makedirs(save_denoise_data_path)
        files_to_copy = ["id2name.json", "id2name4Rec.json", "item_embedding.pt"]
        for file_name in files_to_copy:
            source_path = os.path.join(noise_dataset_path, file_name)
            destination_path = os.path.join(save_denoise_data_path, file_name)
            shutil.copy(source_path, destination_path)

    id2name_path = os.path.join(save_denoise_data_path, "id2name4Rec.json")
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

    model = PeftModel.from_pretrained(model, denoise_model_path, torch_dtype=torch.bfloat16)
    model.merge_and_unload()
    model.eval()

    first_token_ids_list = generate_first_tokenid(tokenizer, id2name_dict)
    id2first_token_id_dict = dict(zip(id2name_dict.keys(), first_token_ids_list))

    data_root = f"{denoise_data}_{sample}" if sample != -1 else denoise_data
    data_root_list = [data_root]

    for file_name in data_root_list:
        test_flag = True if "test" in file_name else False

        data_path = os.path.join(noise_dataset_path, f"{file_name}.csv")
        df = pd.read_csv(data_path)

        df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
        item_ids_list = df["item_ids"].tolist()
        if test_flag:
            item_ids_list = [item_ids[:-1] for item_ids in item_ids_list]

        prompt_list = []
        token_ids_list = []

        for item_ids in tqdm(item_ids_list):
            prompt_list.append(generate_noise_prompt(item_ids, id2name_dict))
            token_ids_list.append([id2first_token_id_dict[item_id] for item_id in item_ids])

        data_dict = {"prompts": prompt_list, "token_ids": token_ids_list}

        with accelerator.split_between_processes(data_dict) as data:
            with torch.no_grad():
                noise_prob_list = []
                for i in tqdm(range((len(data["prompts"]) + batch_size - 1) // batch_size)):
                    first_token_ids = data["token_ids"][i * batch_size : (i + 1) * batch_size]

                    prompts = data["prompts"][i * batch_size : (i + 1) * batch_size]
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
                        accelerator.device
                    )

                    outputs = model(**inputs)
                    logits = outputs.logits
                    noise_prob = get_noise_items_prob(first_token_ids, logits)
                    noise_prob_list.extend(noise_prob)

        noise_prob_list = gather_object(noise_prob_list)
        assert len(noise_prob_list) == len(item_ids_list)

        if accelerator.is_main_process:
            df["noise_prob"] = noise_prob_list
            csv_save_path = os.path.join(save_denoise_data_path, f"{file_name}.csv")
            df.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
