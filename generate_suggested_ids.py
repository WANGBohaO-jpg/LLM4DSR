import fire
import pandas as pd
import pdb
import json
import os
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def batch(list, batch_size):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i : batch_size * (i + 1)]


def grounding(nonexistent_titles, model, tokenizer, item_embedding_table, batch_size):
    with torch.no_grad():
        predict_embeddings = []
        for batch_input in tqdm(
            batch(nonexistent_titles, batch_size=batch_size), total=len(nonexistent_titles) // batch_size + 1
        ):
            inputs = tokenizer(batch_input, return_tensors="pt", padding=True).to("cuda")
            outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            predict_embeddings.append(hidden_states[-1][:, -1, :].detach())
        predict_embeddings = torch.cat(predict_embeddings, dim=0)  # len(nonexistent_titles) x 4096
        
    # item_embedding_table = item_embedding_table.cpu()

    dist = torch.cdist(predict_embeddings, item_embedding_table, p=2)  # 5000 x item_num
    dist[:, 0] = float("inf")
    _, min_indices = torch.min(dist, dim=1)
    
    min_indices_list = min_indices.tolist()
    title_to_index = dict(zip(nonexistent_titles, min_indices_list))

    return title_to_index


def add_suggest_item_ids(df, model, tokenizer, item_embedding_table, name2id_dict, batch_size):
    nonexistent_titles = []
    cnt = 0
    for _, row in df.iterrows():
        for item in eval(row["suggest_item_titles"]):
            if item not in name2id_dict.keys() and item not in nonexistent_titles:
                cnt += 1
                nonexistent_titles.append(item)
    print(f"The number of nonexist titles is {cnt}")

    if len(nonexistent_titles) == 0:
        merged_dict = name2id_dict
    else:
        name2id_dict_2 = grounding(nonexistent_titles, model, tokenizer, item_embedding_table, batch_size)
        merged_dict = {**name2id_dict, **name2id_dict_2}

    suggest_item_ids_list = []
    for _, row in df.iterrows():
        suggest_item_titles = eval(row["suggest_item_titles"])
        suggest_item_ids = [merged_dict[item_title] for item_title in suggest_item_titles]
        suggest_item_ids_list.append(suggest_item_ids)

    df["suggest_item_ids"] = suggest_item_ids_list
    return df


def main(
    data_path: str,
    batchsize: int = 16,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
):
    json_file_path = os.path.join(data_path, "id2name.json")
    with open(json_file_path, "r") as file:
        id2name_dict = json.load(file)
    name2id_dict = {v: int(k) for k, v in id2name_dict.items()}

    item_embedding_path = os.path.join(data_path, "item_embedding.pt")
    item_embedding_table = torch.load(item_embedding_path)
    
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
    model.eval()

    for file_name in os.listdir(data_path):
        if file_name.endswith(".csv"):
            print(f"Begin to process {file_name}")

            file_path = os.path.join(data_path, file_name)
            df = pd.read_csv(file_path)

            if not "suggest_item_titles" in df.columns:
                print("No suggest_item_titles column in the csv file, skip it.")
                continue
                
            # if "suggest_item_ids" in df.columns:
            #     print("suggest_item_ids column already exists in the csv file, skip it.")
            #     continue

            df = add_suggest_item_ids(df, model, tokenizer, item_embedding_table, name2id_dict, batchsize)
            df.to_csv(file_path, index=False)
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
