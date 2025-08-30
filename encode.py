"""
CUDA_VISIBLE_DEVICES=1 nohup python encode.py --dataset_name "Movie_noise_user0.0" --batch_size 64 &
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
import os
from tqdm import tqdm
import argparse


parse = argparse.ArgumentParser()
parse.add_argument("--dataset_name", type=str, default="amazon_game")
parse.add_argument("--batch_size", type=int, default=16)
parse.add_argument("--base_model", type=str, default="/c23034/wbh/Llama3_Checkpoints/")
args = parse.parse_args()

base_model = args.base_model
if "Llama3" in base_model:
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
elif "Llama2" in base_model:
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
model.eval()

save_item_embedding_path = os.path.join("./data/", args.dataset_name, "item_embedding.pt")
# id2name_path = os.path.join("./data/", args.dataset_name, "id2name.json")
id2name_path = os.path.join("./data/", args.dataset_name, "id2name4Rec.json")

# 按照item的连续编号进行排序
with open(id2name_path, "r", encoding="utf-8") as f:
    id2name = json.load(f)
# item_names = [value.strip('"\n').strip(" ") for _, value in sorted(id2name.items())]
item_cids = [int(k) for k in id2name.keys()]
item_names = [v.strip('"\n').strip(" ") for v in id2name.values()]
item_max_id = max(item_cids)


def batch(list, batch_size):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i : batch_size * (i + 1)]


with torch.no_grad():
    all_text = item_names
    # all_embeddings = []
    all_embeddings = torch.zeros((item_max_id + 1, 4096), device="cuda", dtype=torch.bfloat16)
    for batch_cid, batch_text in tqdm(
        zip(batch(item_cids, args.batch_size), batch(all_text, args.batch_size)),
        total=len(all_text) // args.batch_size + 1,
    ):
        batch_cid = torch.tensor(batch_cid, device="cuda")
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        all_embeddings[batch_cid] = hidden_states[-1][:, -1, :].detach()
        # all_embeddings.append(hidden_states[-1][:, -1, :].detach())
    # all_embeddings = torch.cat(all_embeddings, dim=0)

print("================all_embeddings================\n", all_embeddings.shape)
has_nan = torch.isnan(all_embeddings).any().item()
if has_nan:
    print("llm_all_emb contains NaN values.")
    exit(-1)
torch.save(all_embeddings, save_item_embedding_path)
